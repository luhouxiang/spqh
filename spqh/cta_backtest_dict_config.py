# -*- coding: utf-8 -*-
"""
CTP 回测脚本（字典配置版，无命令行）
==================================
基于：vn.py 4.1.0 + vnpy_ctastrategy

特点：
- 用一个 Python 字典 CONFIG 配好所有参数（合约、CSV、费率、乘数等），无需命令行。
- 自动将 CSV 导入 vn.py 数据库（缺失 volume 列会自动补 0）。
- 自动回测：加载数据 → 执行 → 绩效统计 → 画图。
- 如未指定 start/end，会根据 CSV 的时间自动取最早/最晚日期。
- 如未指定 interval，会根据 CSV 频率自动猜测为 1m/5m/15m/30m/60m/1d。

使用方法：
1) 安装依赖：
   pip install "vnpy==4.1.0"
   pip install vnpy-ctastrategy
2) 修改下方 CONFIG 和 TARGET（示例已给出 agl9）
3) 运行：python cta_backtest_dict_config.py

注意：
- CSV 至少包含：datetime, open, high, low, close（volume 若缺失自动补 0）
- CSV 时间建议：分钟/小时：YYYY-MM-DD HH:MM:SS；日线：YYYY-MM-DD
"""
from __future__ import annotations
import matplotlib
matplotlib.use("QtAgg")   # 强制使用 Qt 图形后端（弹窗）
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, Optional, List
from datetime import date, datetime
import numpy as np
import pandas as pd

def setup_matplotlib_backend(prefer_gui: bool = True) -> bool:
    import matplotlib
    if os.environ.get("MPLBACKEND"):
        backend = os.environ["MPLBACKEND"].lower()
        return backend not in ("agg", "pdf", "svg", "ps", "cairo")

    has_display = (sys.platform.startswith("win")
                   or sys.platform == "darwin"
                   or bool(os.environ.get("DISPLAY")))

    if prefer_gui and has_display:
        try:
            import PySide6  # noqa: F401
            matplotlib.use("QtAgg")
            return True
        except Exception:
            try:
                import PyQt5  # noqa: F401
                matplotlib.use("QtAgg")
                return True
            except Exception:
                pass
        try:
            import tkinter  # noqa: F401
            matplotlib.use("TkAgg")
            return True
        except Exception:
            pass

    matplotlib.use("Agg")
    return False

INTERACTIVE = setup_matplotlib_backend(prefer_gui=True)

try:
    from vnpy_ctastrategy import CtaTemplate
    from vnpy_ctastrategy.backtesting import BacktestingEngine
except Exception as e:
    raise ImportError(
        "未找到 vnpy_ctastrategy，请先安装：\n"
        "    pip install vnpy-ctastrategy\n"
        f"原始错误：{e}"
    )

from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import BarData
from vnpy.trader.database import get_database
from vnpy.trader.utility import ArrayManager
import matplotlib.pyplot as plt

import os
from common.logging_cfg import SysLogInit
from cfg import g_cfg
from common.config import DATA_PATH

# =========================
# 策略示例：双均线交叉
# =========================
class DoubleMaStrategy(CtaTemplate):
    author = "you"

    fast_window: int = 10
    slow_window: int = 30
    fixed_size: int = 1

    fast_ma: float = 0.0
    slow_ma: float = 0.0

    parameters = ["fast_window", "slow_window", "fixed_size"]
    variables = ["fast_ma", "slow_ma"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.am = ArrayManager(size=max(self.fast_window, self.slow_window) + 50)

    def on_init(self) -> None:
        self.write_log("策略初始化")
        init_bars = max(self.fast_window, self.slow_window) + 50
        self.load_bar(init_bars)

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        self.write_log("策略停止")

    def on_bar(self, bar: BarData) -> None:
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        fast_arr = self.am.sma(self.fast_window, array=True)
        slow_arr = self.am.sma(self.slow_window, array=True)

        self.fast_ma = float(fast_arr[-1])
        self.slow_ma = float(slow_arr[-1])

        cross_up = (fast_arr[-2] <= slow_arr[-2]) and (fast_arr[-1] > slow_arr[-1])
        cross_dn = (fast_arr[-2] >= slow_arr[-2]) and (fast_arr[-1] < slow_arr[-1])

        if self.pos == 0:
            if cross_up:
                self.buy(bar.close_price, self.fixed_size)
            elif cross_dn:
                self.short(bar.close_price, self.fixed_size)
        elif self.pos > 0 and cross_dn:
            self.sell(bar.close_price, abs(self.pos))
            self.short(bar.close_price, self.fixed_size)
        elif self.pos < 0 and cross_up:
            self.cover(bar.close_price, abs(self.pos))
            self.buy(bar.close_price, self.fixed_size)

        self.put_event()


# =========================
# 工具：频率猜测、Interval 解析
# =========================
def guess_interval_from_csv(dt_series: pd.Series) -> Interval:
    s = pd.to_datetime(dt_series, errors="coerce").dropna().sort_values().unique()
    if len(s) < 3:
        return Interval.DAILY  # 默认给日线
    diffs = [(pd.Timestamp(s[i+1]) - pd.Timestamp(s[i])).total_seconds() for i in range(len(s)-1)]
    if not diffs:
        return Interval.DAILY
    m = median(diffs)
    if abs(m - 60) <= 5:
        return Interval.MINUTE
    if abs(m - 300) <= 20:
        return Interval.MINUTE  # 依然是分钟，建议外部先重采样
    if abs(m - 900) <= 40:
        return Interval.MINUTE
    if abs(m - 1800) <= 60:
        return Interval.MINUTE
    if abs(m - 3600) <= 120:
        return Interval.HOUR
    # 其余大概率为日线
    return Interval.DAILY


# =========================
# CSV → DB 导入
# =========================
def import_csv_to_db(
    csv_path: Path,
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    tz: Optional[str] = "Asia/Shanghai",
) -> tuple[datetime, datetime]:
    """
    必要列：datetime, open, high, low, close
    如果缺失 volume 列，会自动补 0。
    返回：(start_datetime, end_datetime) 便于回测参数缺省时自动填充
    """
    assert csv_path.exists(), f"CSV 不存在：{csv_path}"

    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    need_base = {"datetime", "open", "high", "low", "close"}
    missing = need_base - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}")

    if "volume" not in df.columns:
        df["volume"] = 0

    if "open_interest" not in df.columns:
        df["open_interest"] = 0

    dt = pd.to_datetime(df["datetime"], errors="raise")
    if tz:
        # 本地化到 tz，再转为 naive（无时区）
        try:
            dt = dt.dt.tz_localize(tz).dt.tz_convert(None)
        except Exception:
            # 已有时区或不能本地化时，直接去 tzinfo
            dt = dt.dt.tz_convert(None) if hasattr(dt.dt, "tz_convert") else dt

    bars: List[BarData] = []
    for i, row in df.iterrows():
        bars.append(
            BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=dt.iloc[i].to_pydatetime(),
                interval=interval,
                volume=float(row["volume"]),
                turnover=0.0,
                open_interest=float(row.get("open_interest", 0)),
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                gateway_name="DB",
            )
        )

    db = get_database()
    db.save_bar_data(bars)

    start_dt = dt.min().to_pydatetime()
    end_dt = dt.max().to_pydatetime()
    print(f"已导入 {len(bars)} 条 {interval.name} K 线到数据库：{symbol}.{exchange.value}")
    print(f"数据时间：{start_dt} ~ {end_dt}")
    return start_dt, end_dt


def json_default(o):
    if isinstance(o, (date, datetime)):     # 日期/时间 → ISO 字符串
        return o.isoformat()
    if isinstance(o, (np.integer,)):        # numpy 整数 → int
        return int(o)
    if isinstance(o, (np.floating,)):       # numpy 浮点 → float
        return float(o)
    if isinstance(o, (np.bool_,)):          # numpy 布尔 → bool
        return bool(o)
    return str(o)                           # 兜底方案（极少触发）

# =========================
# 回测主流程
# =========================
def run_backtest_and_maybe_show(
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    end: datetime,
    rate: float,
    slippage: float,
    size: int,
    pricetick: float,
    capital: float,
    strategy_params: dict,
    out_dir: Path,
) -> None:
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=interval,
        start=start,
        end=end,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
    )
    engine.add_strategy(DoubleMaStrategy, strategy_params)

    print("加载数据...")
    engine.load_data()

    print("开始回测...")
    engine.run_backtesting()

    print("计算绩效...")
    df = engine.calculate_result()
    stats = engine.calculate_statistics()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backtest_timeseries.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=json_default)

    try:
        plt.figure(figsize=(10, 5))
        if "balance" in df.columns:
            plt.plot(df["balance"])
            plt.title("Backtest Balance")
            plt.xlabel("Index")
            plt.ylabel("Balance")
            plt.tight_layout()
            plt.savefig(out_dir / "balance.png", dpi=150)
        plt.close()
        print(f"已保存：{out_dir/'balance.png'}")
    except Exception as e:
        print(f"保存图像失败：{e}")

    print("\n===== 绩效指标（关键项） =====")
    for k in ["start_date", "end_date", "total_days", "end_balance", "max_drawdown", "max_ddpercent",
              "total_net_pnl", "total_return", "annual_return", "sharpe_ratio"]:
        if k in stats:
            print(f"{k}: {stats[k]}")

    if INTERACTIVE:
        print("\n检测到可用 GUI，弹出交互式图表窗口...")
        engine.show_chart()
    else:
        print("\n未检测到 GUI（或后端为 Agg），已将结果保存至：")
        print(f"  - {out_dir/'stats.json'}")
        print(f"  - {out_dir/'backtest_timeseries.csv'}")
        print(f"  - {out_dir/'balance.png'}")


# 字典配置（将命令行参数改为字典映射）
# =========================
CONFIG: Dict[str, dict] = {
    # 示例：你的 agl9（已按你上传的CSV推断为日线）
    "agl9": {
        "csv": "spqhagl9.csv",   # 或者填写 spqhagl9.csv（程序会自动补 volume=0）
        "exchange": "SHFE",                   # 交易所
        "interval": "1d",                     # 1m/60m/1d；留空则自动根据 CSV 频率猜测
        # "start": "2012-05-10",              # 可留空自动用 CSV 最早日期
        # "end": "2013-03-11",                # 可留空自动用 CSV 最晚日期
        "rate": 0.00025,
        "slippage": 1,
        "size": 15,                           # AG 合约常见乘数：15
        "pricetick": 1,                       # AG tick：1
        "capital": 1_000_000,
        "tz": "Asia/Shanghai",
        "strategy": {"fast_window": 10, "slow_window": 30, "fixed_size": 1},
    },
    # 也可以在此继续添加其它标的映射……
}

# 选择要运行的标的键
TARGET = "agl9"


# =========================
# 主流程（不使用命令行）
# =========================
def main() -> None:
    SysLogInit('cta_backtest_dict_config.log', "logs")
    here = Path(__file__).resolve().parent
    cfg = CONFIG.get(TARGET)
    assert cfg, f"未在 CONFIG 中找到标的：{TARGET}"

    symbol = TARGET
    exchange = Exchange(str(cfg.get("exchange", "SHFE")).upper())

    # CSV 路径优先找配置文件名（相对脚本目录），其次允许你给绝对路径
    csv_name = cfg.get("csv")
    assert csv_name, f"请在 CONFIG['{TARGET}']['csv'] 填写CSV文件名或绝对路径"
    csv_path = os.path.join(DATA_PATH, "Exports", csv_name)
    csv_path = Path(csv_path)
    if not csv_path.exists():
        # 尝试脚本目录下查找
        csv_path = (here / csv_name).resolve()

    assert csv_path.exists(), f"CSV 文件不存在：{csv_path}"

    # 读CSV，准备时间与频率
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    assert "datetime" in df.columns, "CSV 必须包含 datetime 列"
    dt_series = pd.to_datetime(df["datetime"], errors="raise")

    # interval
    interval_text = str(cfg.get("interval", "")).strip().lower()
    if interval_text in ("1d", "d", "day", "daily"):
        interval = Interval.DAILY
    elif interval_text in ("60m", "1h", "h", "hour"):
        interval = Interval.HOUR
    elif interval_text in ("1m", "m", "minute"):
        interval = Interval.MINUTE
    else:
        interval = guess_interval_from_csv(dt_series)

    # start/end 若未提供，从 CSV 推断
    start_str = cfg.get("start")
    end_str = cfg.get("end")
    if start_str and end_str:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    else:
        start_dt = pd.to_datetime(dt_series.min()).to_pydatetime()
        end_dt = pd.to_datetime(dt_series.max()).to_pydatetime()

    # 其它参数
    rate = float(cfg.get("rate", 2.5/10000))
    slippage = float(cfg.get("slippage", 1))
    size = int(cfg.get("size", 10))
    pricetick = float(cfg.get("pricetick", 1))
    capital = float(cfg.get("capital", 1_000_000))
    tz = cfg.get("tz", "Asia/Shanghai")
    strategy_params = dict(cfg.get("strategy", {}))

    vt_symbol = f"{symbol}.{exchange.value}"
    out_dir = here / "backtest_output" / vt_symbol.replace(".", "_")

    print("=== 配置汇总 ===")
    print(f"  vt_symbol : {vt_symbol}")
    print(f"  csv       : {csv_path}")
    print(f"  interval  : {interval.name}")
    print(f"  start~end : {start_dt} ~ {end_dt}")
    print(f"  rate/slip : {rate} / {slippage}")
    print(f"  size/tick : {size} / {pricetick}")
    print(f"  capital   : {capital}")
    print(f"  strategy  : {strategy_params}")
    print(f"  输出目录  : {out_dir}")
    print(f"  GUI可用?  : {INTERACTIVE}")

    import_csv_to_db(
        csv_path=csv_path,
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        tz=tz,
    )
    run_backtest_and_maybe_show(
        vt_symbol=vt_symbol,
        interval=interval,
        start=start_dt,
        end=end_dt,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
        strategy_params=strategy_params,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
