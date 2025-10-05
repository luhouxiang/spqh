# utils_backtest.py
import os, sys, json
from pathlib import Path
import ast
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
    idx = pd.to_datetime(df.index)
    df = df.assign(
        datetime=idx
    )
    (out_dir/"backtest_timeseries.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    with open(out_dir/"stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=json_default)

    try:
        plt.figure(figsize=(10,5))
        fw = int(strategy_params.get("fast_window", 10))
        sw = int(strategy_params.get("slow_window", 30))
        _save_price_trades_and_indicators_combined(df=df, out_dir=out_dir, fast_window=fw, slow_window=sw)

        if "balance" in df.columns:
            plt.plot(df["balance"]); plt.title("Backtest Balance")
            plt.xlabel("Index"); plt.ylabel("Balance"); plt.tight_layout()
            plt.savefig(out_dir/"balance.png", dpi=150)
        plt.close()
        print(f"已保存：{out_dir/'balance.png'}")
    except Exception as e:
        print(f"保存图像失败：{e}")

        # # —— 调用：用策略窗口作为默认指标窗口（与 balance.png 同位置输出）
        # try:
        #     fw = int(strategy_params.get("fast_window", 10))
        #     sw = int(strategy_params.get("slow_window", 30))
        #     _save_price_trades_and_indicators_combined(df=df, out_dir=out_dir, fast_window=fw, slow_window=sw)
        # except Exception as e:
        #     print(f"[WARN] 生成“价格+指标合并图”失败：{e}")

    if interactive:
        print("检测到 GUI：弹出交互式图表窗口...")
        engine.show_chart()
    else:
        print("未检测到 GUI：结果已保存到：", out_dir)

    return df, stats


def _parse_trades_cell(cell):
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s in ("[]", "{}"):
        return []
    try:
        obj = ast.literal_eval(s)
    except Exception:
        try:
            obj = ast.literal_eval(s.replace("null", "None"))
        except Exception:
            return []
    if isinstance(obj, dict):  # 单条成交
        return [obj]
    if isinstance(obj, list):
        return obj
    return []

def _classify_trade(tr):
    d = str(tr.get("direction", "")).lower()
    o = str(tr.get("offset", "")).lower()
    long_like  = any(k in d for k in ["long", "buy", "多"])
    short_like = any(k in d for k in ["short", "sell", "空"])
    open_like  = (o == "" or "open" in o or "开" in o)         # 开仓
    close_like = ("close" in o or "平" in o)                   # 平仓
    if long_like and open_like:   return "open_long"
    if long_like and close_like:  return "close_long"
    if short_like and open_like:  return "open_short"
    if short_like and close_like: return "close_short"
    return "unknown"

def _extract_trade_time(tr, fallback_ts):
    import pandas as pd
    for k in ["time", "datetime", "dt", "trade_time", "ts"]:
        if k in tr and tr[k]:
            try:
                return pd.to_datetime(tr[k])
            except Exception:
                pass
    return fallback_ts

def _save_price_trades_and_indicators_combined(df, out_dir: Path, fast_window=10, slow_window=30):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ① 决定横轴（改成下面这行）
    x, x_label = _pick_time_axis(df, out_dir)

    # # 时间轴：优先 DatetimeIndex / datetime列，否则用序号
    # if isinstance(df.index, pd.DatetimeIndex):
    #     x = df.index; x_label = "datetime"
    # elif "datetime" in df.columns:
    #     x = pd.to_datetime(df["datetime"], errors="coerce"); x_label = "datetime"
    # else:
    #     x = np.arange(len(df)); x_label = "bar_index"

    # 价格列
    price_col = next((c for c in ["close_price","close","Close","last_price","price"] if c in df.columns), None)
    if price_col is None:
        print("[WARN] 未找到价格列，跳过合并图生成"); return

    # 解析成交，分类四类点
    points = {k: [] for k in ["open_long","close_long","open_short","close_short"]}
    if "trades" in df.columns:
        for i, row in df.iterrows():
            trs = _parse_trades_cell(row["trades"])
            if not trs: continue
            fallback = x[i] if hasattr(x, "__len__") else None
            for tr in trs:
                cat = _classify_trade(tr)
                if cat in points:
                    px = tr.get("price", row.get(price_col, np.nan))
                    ts = _extract_trade_time(tr, fallback)
                    if isinstance(x, (pd.Series, pd.DatetimeIndex)) and isinstance(ts, pd.Timestamp):
                        x_val = ts
                    else:
                        x_val = fallback if fallback is not None else i
                    points[cat].append((x_val, px))

    # 若没有 trades，则用持仓变化做近似（可留空或保留）
    if not any(points.values()):
        start_col = next((c for c in df.columns if c.lower() in ("start_pos","start_position","pos_start")), None)
        end_col   = next((c for c in df.columns if c.lower() in ("end_pos","end_position","pos_end")), None)
        if start_col and end_col:
            spos, epos = df[start_col].values, df[end_col].values
            for i in range(len(df)):
                if epos[i] > spos[i]:
                    points["open_long"].append((x[i], df.iloc[i][price_col]))
                elif epos[i] < spos[i]:
                    points["open_short"].append((x[i], df.iloc[i][price_col]))

    # 指标：优先 fast_ma/slow_ma，否则按窗口（默认 10/30）计算
    if ("fast_ma" in df.columns) and ("slow_ma" in df.columns):
        fast_ma = df["fast_ma"]; slow_ma = df["slow_ma"]
    else:
        close_ser = pd.Series(df[price_col].values)
        fast_ma = close_ser.rolling(window=int(fast_window), min_periods=1).mean()
        slow_ma = close_ser.rolling(window=int(slow_window), min_periods=1).mean()

    # 绘制两个面板 → 拼成一张（不改前端调用）
    price_panel_path = out_dir / "_panel_price.png"
    indi_panel_path  = out_dir / "_panel_indicators.png"
    final_path       = out_dir / "price_indicators_combined.png"

    # 面板A：价格 + 交易标记
    plt.figure(figsize=(12, 6))
    plt.plot(x, df[price_col], label=price_col)
    if points["open_long"]:
        xs, ys = zip(*points["open_long"]);  plt.scatter(xs, ys, marker="^", s=40, label="Open Long")
    if points["close_long"]:
        xs, ys = zip(*points["close_long"]); plt.scatter(xs, ys, marker="v", s=40, label="Close Long")
    if points["open_short"]:
        xs, ys = zip(*points["open_short"]); plt.scatter(xs, ys, marker="x", s=40, label="Open Short")
    if points["close_short"]:
        xs, ys = zip(*points["close_short"]); plt.scatter(xs, ys, marker="o", s=30, label="Close Short")
    plt.title("Price with Trade Markers")
    plt.xlabel(x_label); plt.ylabel(price_col); plt.legend(); plt.tight_layout()
    plt.savefig(price_panel_path, dpi=150); plt.close()

    # 面板B：指标
    plt.figure(figsize=(12, 3.5))
    plt.plot(x, fast_ma, label="fast_ma")
    plt.plot(x, slow_ma, label="slow_ma")
    plt.title("Indicators (SMA)")
    plt.xlabel(x_label); plt.ylabel("value"); plt.legend(); plt.tight_layout()
    plt.savefig(indi_panel_path, dpi=150); plt.close()

    # 叠成一张图（需要 Pillow）
    try:
        from PIL import Image
        top = Image.open(price_panel_path); bot = Image.open(indi_panel_path)
        w = min(top.width, bot.width)
        if top.width != w: top = top.resize((w, int(top.height * w / top.width)))
        if bot.width != w: bot = bot.resize((w, int(bot.height * w / bot.width)))
        combo = Image.new("RGB", (w, top.height + bot.height), (255, 255, 255))
        combo.paste(top, (0, 0)); combo.paste(bot, (0, top.height))
        combo.save(final_path, format="PNG")
        print(f"已生成合并图：{final_path}")
    except Exception as e:
        print(f"[WARN] 合并图生成失败（可安装 pillow）：{e}")


def _pick_time_axis(df, out_dir: Path):
    import pandas as pd
    import numpy as np

    # 1) 优先：DataFrame 的索引是时间
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index, "datetime"

    # 2) 其次：DataFrame 有 datetime 列
    if "datetime" in df.columns:
        t = pd.to_datetime(df["datetime"], errors="coerce")
        if t.notna().any():
            return t, "datetime"

    # 3) 再次：从 CSV 里读取 datetime 列
    ts_csv = out_dir / "backtest_timeseries.csv"
    if ts_csv.exists():
        tmp = pd.read_csv(ts_csv)
        if "datetime" in tmp.columns:
            t = pd.to_datetime(tmp["datetime"], errors="coerce")
            if len(t) == len(df) and t.notna().any():
                return t, "datetime"

    # 4) 兜底：用序号
    return np.arange(len(df)), "bar_index"
