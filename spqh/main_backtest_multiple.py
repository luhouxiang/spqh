# main_backtest_multi.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from vnpy.trader.constant import Interval, Exchange

# 仍然复用你已有的配置/工具/策略
from settings import CONFIG, DATA_PATH, EXPORTS_SUBDIR, OUTPUT_ROOT, TIMEZONE
from strategies.double_ma import DoubleMaStrategy
from utils_backtest import (
    setup_matplotlib_backend,
    guess_interval_from_csv,
    import_csv_to_db,
    run_backtest_and_output,   # 要求该函数返回 (df, stats)
)


# =============== 单标的回测（原来 main() 的逻辑提炼） ===============
def run_one_symbol(symbol_key: str) -> Tuple[pd.DataFrame, dict, Path]:
    cfg = CONFIG[symbol_key]
    symbol = symbol_key
    exchange = Exchange(cfg.get("exchange", "SHFE").upper())
    csv_name = cfg["csv"]

    # CSV 路径：优先 data/Exports，其次脚本目录
    here = Path(__file__).parent
    csv_path = (DATA_PATH / csv_name)
    if not csv_path.exists():
        csv_path = (here / csv_name)
    csv_path = csv_path.resolve()
    assert csv_path.exists(), f"CSV 不存在：{csv_path}"

    # 推断 interval 与起止
    df_raw = pd.read_csv(csv_path)
    df_raw.columns = [c.lower().strip() for c in df_raw.columns]
    dt_series = pd.to_datetime(df_raw["datetime"], errors="raise")

    txt = str(cfg.get("interval", "")).lower()
    if txt in ("1d", "d", "day", "daily"):
        interval = Interval.DAILY
    elif txt in ("60m", "1h", "h", "hour"):
        interval = Interval.HOUR
    elif txt in ("1m", "m", "minute"):
        interval = Interval.MINUTE
    else:
        interval = guess_interval_from_csv(dt_series)

    if cfg.get("start") and cfg.get("end"):
        from datetime import datetime
        start_dt = datetime.strptime(cfg["start"], "%Y-%m-%d")
        end_dt = datetime.strptime(cfg["end"], "%Y-%m-%d")
    else:
        start_dt = pd.to_datetime(dt_series.min()).to_pydatetime()
        end_dt = pd.to_datetime(dt_series.max()).to_pydatetime()

    # 其它参数
    rate = float(cfg.get("rate", 2.5 / 10000))
    slippage = float(cfg.get("slippage", 1))
    size = int(cfg.get("size", 10))
    pricetick = float(cfg.get("pricetick", 1))
    capital = float(cfg.get("capital", 1_000_000))
    params = dict(cfg.get("strategy", {}))

    vt_symbol = f"{symbol}.{exchange.value}"
    out_dir = (OUTPUT_ROOT / vt_symbol.replace(".", "_")).resolve()

    print("=== 配置汇总 ===")
    print(
        f"vt_symbol={vt_symbol}  csv={csv_path}\n"
        f"interval={interval.name}  start~end={start_dt} ~ {end_dt}\n"
        f"rate/slip={rate}/{slippage}  size/tick={size}/{pricetick}  capital={capital}\n"
        f"strategy={params}\n输出目录={out_dir}\n"
    )

    # 导入数据库 + 回测 + 输出（复用你现有工具函数）
    import_csv_to_db(csv_path, symbol, exchange, interval, tz=TIMEZONE)
    df, stats = run_backtest_and_output(
        strategy_cls=DoubleMaStrategy,
        vt_symbol=vt_symbol,
        interval=interval,
        start_dt=start_dt,
        end_dt=end_dt,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
        strategy_params=params,
        out_dir=out_dir,
        interactive=False,  # 多标的批量时一般不弹窗
    )
    # 要求 run_backtest_and_output 返回 (df, stats)。如果你当前版本没返回，简单改一下它，在函数末尾 `return df, stats`

    # 给组合层需要的信息：df（含 net_pnl/balance，index 为时间）和 out_dir
    return df, stats, out_dir


# =============== 组合层：把多个标的的净值/收益合成一条曲线 ===============
def build_portfolio_curve(results: List[Tuple[str, pd.DataFrame, dict]]) -> Path:
    """
    results: [(symbol_key, df, stats), ...]
    将各标的的 net_pnl 对齐相加，得到组合净值（用各标的 capital 之和为初始资金）
    """
    import matplotlib.pyplot as plt

    # 1) 汇总各标的 net_pnl 序列（按时间对齐）
    pnl_frames = []
    capitals = 0.0
    for symbol_key, df, stats in results:
        s = df["net_pnl"].copy()
        s.name = symbol_key
        pnl_frames.append(s)
        capitals += float(stats.get("capital", 0.0) or 0.0) if "capital" in stats else 0.0
    pnl_df = pd.concat(pnl_frames, axis=1).fillna(0.0)   # 对齐缺失补0表示无交易

    # 2) 组合净 PnL（逐时点求和）与组合资金曲线
    portfolio_net = pnl_df.sum(axis=1)
    if capitals <= 0:   # 若 stats 里没有 capital，就从 CONFIG 汇总
        capitals = sum(float(CONFIG[k].get("capital", 0.0)) for k, _, _ in results)
    portfolio_balance = capitals + portfolio_net.cumsum()

    # 3) 保存到专属目录
    keys = [k for k, _, _ in results]
    port_dir = (OUTPUT_ROOT / f"portfolio_{'_'.join(keys)}").resolve()
    port_dir.mkdir(parents=True, exist_ok=True)

    # 明细 CSV
    out_ts = pd.DataFrame({
        "portfolio_net_pnl": portfolio_net,
        "portfolio_balance": portfolio_balance
    })
    out_ts.to_csv(port_dir / "portfolio_timeseries.csv", index=True, index_label="datetime")

    # 组合净值图
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_balance)
    plt.title("Portfolio Balance")
    plt.xlabel("datetime")
    plt.ylabel("Balance")
    plt.tight_layout()
    plt.savefig(port_dir / "portfolio_balance.png", dpi=150)
    plt.close()

    print(f"组合输出目录：{port_dir}")
    return port_dir


def main():
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


if __name__ == "__main__":
    main()
