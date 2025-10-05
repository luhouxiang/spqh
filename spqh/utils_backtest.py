# utils_backtest.py
import os, sys, json
from pathlib import Path
from datetime import date, datetime
from statistics import median
from typing import Optional, List, Type

import numpy as np
import pandas as pd
import matplotlib


def setup_matplotlib_backend(prefer_gui: bool = True) -> bool:
    """返回 True=可交互（弹窗），False=无头（仅保存文件）"""
    if os.environ.get("MPLBACKEND"):
        b = os.environ["MPLBACKEND"].lower()
        return b not in ("agg", "pdf", "svg", "ps", "cairo")

    has_display = (sys.platform.startswith("win")
                   or sys.platform == "darwin"
                   or bool(os.environ.get("DISPLAY")))

    if prefer_gui and has_display:
        try:
            import PySide6  # noqa
            matplotlib.use("QtAgg")
            return True
        except Exception:
            try:
                import PyQt5  # noqa
                matplotlib.use("QtAgg")
                return True
            except Exception:
                pass
        try:
            import tkinter   # noqa
            matplotlib.use("TkAgg")
            return True
        except Exception:
            pass

    matplotlib.use("Agg")
    return False


def guess_interval_from_csv(dt_series: pd.Series):
    from vnpy.trader.constant import Interval
    s = pd.to_datetime(dt_series, errors="coerce").dropna().sort_values().unique()
    if len(s) < 3:
        return Interval.DAILY
    diffs = [(pd.Timestamp(s[i+1]) - pd.Timestamp(s[i])).total_seconds() for i in range(len(s)-1)]
    if not diffs:
        return Interval.DAILY
    m = median(diffs)
    if abs(m-60)<=5: return Interval.MINUTE
    if abs(m-300)<=20: return Interval.MINUTE
    if abs(m-900)<=40: return Interval.MINUTE
    if abs(m-1800)<=60: return Interval.MINUTE
    if abs(m-3600)<=120: return Interval.HOUR
    return Interval.DAILY


def import_csv_to_db(csv_path: Path, symbol: str, exchange, interval, tz: Optional[str] = "Asia/Shanghai"):
    """CSV→DB：补 volume/open_interest，时间本地化→naive，返回 (start_dt, end_dt)"""
    from vnpy.trader.object import BarData
    from vnpy.trader.database import get_database

    assert csv_path.exists(), f"CSV 不存在：{csv_path}"
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    need = {"datetime","open","high","low","close"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"CSV 缺少必要列：{miss}")

    if "volume" not in df.columns:
        df["volume"] = 0
    if "open_interest" not in df.columns:
        df["open_interest"] = 0

    dt = pd.to_datetime(df["datetime"], errors="raise")
    if tz:
        try:
            dt = dt.dt.tz_localize(tz).dt.tz_convert(None)
        except Exception:
            dt = dt.dt.tz_convert(None) if hasattr(dt.dt, "tz_convert") else dt

    bars: List[BarData] = []
    for i, row in df.iterrows():
        bars.append(BarData(
            symbol=symbol, exchange=exchange, datetime=dt.iloc[i].to_pydatetime(),
            interval=interval, volume=float(row["volume"]), turnover=0.0,
            open_interest=float(row.get("open_interest", 0)),
            open_price=float(row["open"]), high_price=float(row["high"]),
            low_price=float(row["low"]),  close_price=float(row["close"]),
            gateway_name="DB"
        ))

    db = get_database()
    db.save_bar_data(bars)
    return dt.min().to_pydatetime(), dt.max().to_pydatetime()


def json_default(o):
    if isinstance(o, (date, datetime)):  return o.isoformat()
    if isinstance(o, (np.integer,)):     return int(o)
    if isinstance(o, (np.floating,)):    return float(o)
    if isinstance(o, (np.bool_,)):       return bool(o)
    return str(o)


def run_backtest_and_output(
    strategy_cls: Type, vt_symbol: str, interval, start_dt, end_dt,
    rate: float, slippage: float, size: int, pricetick: float, capital: float,
    strategy_params: dict, out_dir: Path, interactive: bool
):
    from vnpy_ctastrategy.backtesting import BacktestingEngine
    import matplotlib.pyplot as plt

    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol, interval=interval, start=start_dt, end=end_dt,
        rate=rate, slippage=slippage, size=size, pricetick=pricetick, capital=capital
    )
    engine.add_strategy(strategy_cls, strategy_params)

    print("加载数据..."); engine.load_data()
    print("开始回测..."); engine.run_backtesting()
    print("计算绩效..."); df = engine.calculate_result(); stats = engine.calculate_statistics()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"backtest_timeseries.csv").write_text(df.to_csv(index=True), encoding="utf-8")
    with open(out_dir/"stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=json_default)

    try:
        plt.figure(figsize=(10,5))
        if "balance" in df.columns:
            plt.plot(df["balance"]); plt.title("Backtest Balance")
            plt.xlabel("Index"); plt.ylabel("Balance"); plt.tight_layout()
            plt.savefig(out_dir/"balance.png", dpi=150)
        plt.close()
        print(f"已保存：{out_dir/'balance.png'}")
    except Exception as e:
        print(f"保存图像失败：{e}")

    if interactive:
        print("检测到 GUI：弹出交互式图表窗口...")
        engine.show_chart()
    else:
        print("未检测到 GUI：结果已保存到：", out_dir)

    return df, stats
