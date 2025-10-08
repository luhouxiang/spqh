# main_backtest_multi.py
from __future__ import annotations

from common.logging_cfg import SysLogInit
from cfg import g_cfg

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from vnpy.trader.constant import Interval, Exchange

# 仍然复用你已有的配置/工具/策略
from common.settings import CONFIG, DATA_PATH, EXPORTS_SUBDIR, OUTPUT_ROOT, TIMEZONE
from common.args_from_config import compute_run_args_from_config
from strategies.double_ma import DoubleMaStrategy
from strategies.double_sy import DoubleSyStrategy
from common.utils_backtest import (
    setup_matplotlib_backend,
    guess_interval_from_csv,
    import_csv_to_db,
    run_backtest_and_output,   # 要求该函数返回 (df, stats)
)


# =============== 单标的回测（原来 main() 的逻辑提炼） ===============
# 放在 main_backtest_multiple.py 顶部附近或公共工具模块中

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
from vnpy.trader.constant import Interval, Exchange

# 你的已有依赖
# from settings import CONFIG, DATA_PATH, OUTPUT_ROOT, TIMEZONE
# from utils_backtest import guess_interval_from_csv, import_csv_to_db, run_backtest_and_output
# from double_sy import DoubleSyStrategy


def run_one_symbol(symbol_key: str) -> Tuple[pd.DataFrame, dict, Path]:
    args = compute_run_args_from_config(symbol_key)

    print("=== 配置汇总 ===")
    print(
        f"vt_symbol={args.vt_symbol}  csv={args.csv_path}\n"
        f"interval={args.interval.name}  start~end={args.start_dt} ~ {args.end_dt}\n"
        f"rate/slip={args.rate}/{args.slippage}  size/tick={args.size}/{args.pricetick}  capital={args.capital}\n"
        f"strategy={args.strategy_params}\n输出目录={args.out_dir}\n"
    )

    # 导入数据库（K线） → 回测 → 输出
    import_csv_to_db(args.csv_path, args.symbol, args.exchange, args.interval, tz=TIMEZONE)

    # df, stats = run_backtest_and_output(
    #     strategy_cls=DoubleSyStrategy,
    #     **args.as_kwargs(),        # 直接解包核心参数
    #     interactive=False,         # 批量不弹窗
    # )

    df, stats = run_backtest_and_output(
        strategy_cls=DoubleSyStrategy,
        vt_symbol=args.vt_symbol, interval=args.interval.value, start_dt=args.start_dt, end_dt=args.end_dt,
        rate=args.rate, slippage=args.slippage,
        size=args.size, pricetick=args.pricetick, capital=args.capital,
        strategy_params=args.strategy_params, out_dir=args.out_dir, interactive=False
    )

    return df, stats, args.out_dir


# =============== 组合层：把多个标的的净值/收益合成一条曲线 ===============
def build_portfolio_curve(results: List[Tuple[str, pd.DataFrame, dict]]) -> Path:
    """
    results: [(symbol_key, df, stats), ...]
    将各标的的 net_pnl 对齐相加，得到组合净值。
    兜底策略：
      - 无 net_pnl 用 balance.diff() 推导；
      - net_pnl/balance 都没有 → 用 0 序列（长度=该 df 行数）；
      - df 完全为空（无行） → 跳过该标的。
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    pnl_series_list: List[pd.Series] = []
    capitals = 0.0
    included_keys: List[str] = []

    for symbol_key, df, stats in results:
        # 1) 跳过完全空df（没有任何行）
        if df is None or len(df) == 0:
            print(f"[WARN] {symbol_key}: df为空（无行），跳过。")
            continue

        # 2) 统一索引为时间
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                idx = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.set_index(idx)
            else:
                print(f"[WARN] {symbol_key}: 无 DatetimeIndex 或 datetime 列，跳过。")
                continue
        df = df.sort_index()

        # 3) 安全提取 PnL
        s = None
        if "net_pnl" in df.columns:
            s = pd.to_numeric(df["net_pnl"], errors="coerce").fillna(0.0)
        elif "balance" in df.columns:
            bal = pd.to_numeric(df["balance"], errors="coerce")
            if bal.notna().any():
                # 资本优先取 stats['capital']，否则用首日余额兜底
                cap = stats.get("capital", np.nan) if isinstance(stats, dict) else np.nan
                if not np.isfinite(cap):
                    cap = float(bal.dropna().iloc[0])
                s = bal.diff().fillna(0.0)
                # 首条净Pnl = 首日余额 - 资本
                s.iloc[0] = float(bal.iloc[0]) - float(cap)
        # 3.1 都没有 → 用0序列
        if s is None:
            s = pd.Series(0.0, index=df.index, name=symbol_key)
            print(f"[INFO] {symbol_key}: 缺少 net_pnl/balance，用0序列兜底。")

        # 4) 记录到组合
        s.name = symbol_key
        pnl_series_list.append(s)
        included_keys.append(symbol_key)

        # 资本只对“纳入组合”的标的累加（若 stats 没给就不叠加）
        cap_stat = (stats or {}).get("capital", None)
        if cap_stat is not None:
            try:
                capitals += float(cap_stat)
            except Exception:
                pass

    # 5) 若仍无任何序列，创建空目录返回
    keys_all = [k for k, _, _ in results]
    if not pnl_series_list:
        port_dir = (OUTPUT_ROOT / f"portfolio_{'_'.join(keys_all) if keys_all else 'empty'}").resolve()
        port_dir.mkdir(parents=True, exist_ok=True)
        print(f"[WARN] 无有效标的用于组合，已创建空目录：{port_dir}")
        return port_dir

    # 6) 对齐求和
    pnl_df = pd.concat(pnl_series_list, axis=1).sort_index().fillna(0.0)
    portfolio_net = pnl_df.sum(axis=1)

    # 资本兜底：若没取到，就以0（曲线从净Pnl累计起）
    portfolio_balance = float(capitals) + portfolio_net.cumsum()

    # 7) 输出
    port_dir = (OUTPUT_ROOT / f"portfolio_{'_'.join(included_keys)}").resolve()
    port_dir.mkdir(parents=True, exist_ok=True)

    out_ts = pd.DataFrame({
        "portfolio_net_pnl": portfolio_net,
        "portfolio_balance": portfolio_balance
    })
    out_ts.to_csv(port_dir / "portfolio_timeseries.csv", index=True, index_label="datetime")

    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_balance)
    plt.title("Portfolio Balance")
    plt.xlabel("datetime")
    plt.ylabel("Balance")
    plt.tight_layout()
    plt.savefig(port_dir / "portfolio_balance.png", dpi=150)
    plt.close()

    print(f"组合输出目录：{port_dir}（纳入标的：{included_keys}）")
    return port_dir



def main():
    SysLogInit('main_backtest_multiple.log', "logs")
    g_cfg.load_yaml()   # 默认最先加载配置文件
    logging.info("work begin...")
    # 只初始化一次 GUI/后端（批量时一般不弹窗）
    setup_matplotlib_backend(prefer_gui=False)

    # 你想跑哪些标的：
    # 1) 跑全部：targets = list(CONFIG.keys())
    # 2) 跑部分：targets = ["agl9", "apl9"]
    targets = list(CONFIG.keys())

    all_results: List[Tuple[str, pd.DataFrame, dict]] = []

    for key in targets:
        print(f"\n======== 回测 {key} ========")
        df, stats, out_dir = run_one_symbol(key)
        # 把 capital 塞进 stats，方便组合层统计
        stats = dict(stats)
        stats.setdefault("capital", float(CONFIG[key].get("capital", 0.0)))
        all_results.append((key, df, stats))

    # 组合净值（可选：如果只跑一个，这步自然也能生成“单标的=组合”的曲线）
    build_portfolio_curve(all_results)
    logging.info("work end.")


if __name__ == "__main__":
    main()
