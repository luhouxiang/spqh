from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional, Any

from pathlib import Path
import pandas as pd
from vnpy.trader.constant import Interval, Exchange
from common.config import CONF_PATH, DATA_PATH
from common.settings import CONFIG, TARGET, DATA_PATH, EXPORTS_SUBDIR, OUTPUT_ROOT, TIMEZONE
from common.utils_backtest import guess_interval_from_csv
import os


@dataclass
class RunArgs:
    # ——— 给 run_backtest_and_output 的核心参数 ———
    vt_symbol: str
    interval: Interval
    start_dt: pd.Timestamp
    end_dt: pd.Timestamp
    rate: float
    slippage: float
    size: int
    pricetick: float
    capital: float
    strategy_params: Dict[str, Any]
    out_dir: Path

    # ——— 附带原材料，便于导入/日志等 ———
    csv_path: Path
    symbol: str
    exchange: Exchange

    def as_kwargs(self) -> Dict[str, Any]:
        """便于直接 **args 传给 run_backtest_and_output"""
        return dict(
            vt_symbol=self.vt_symbol,
            interval=self.interval,
            start_dt=self.start_dt.to_pydatetime() if isinstance(self.start_dt, pd.Timestamp) else self.start_dt,
            end_dt=self.end_dt.to_pydatetime() if isinstance(self.end_dt, pd.Timestamp) else self.end_dt,
            rate=self.rate,
            slippage=self.slippage,
            size=self.size,
            pricetick=self.pricetick,
            capital=self.capital,
            strategy_params=self.strategy_params,
            out_dir=self.out_dir,
        )


def compute_run_args_from_config(symbol_key: str) -> RunArgs:
    """
    从 CONFIG[symbol_key] 解析出 run_backtest_and_output 所需的全部参数与相关路径。
    可在任何地方复用。
    """
    cfg = CONFIG[symbol_key]
    symbol = symbol_key
    exchange = Exchange(cfg.get("exchange", "SHFE").upper())

    # 解析 CSV 路径：优先 DATA_PATH 下的文件，其次脚本目录
    csv_name = cfg["csv"]
    here = Path(__file__).parent
    csv_path = Path(os.path.join(DATA_PATH, csv_name))
    if not csv_path.exists():
        csv_path = Path(os.path.join(here, csv_name))
    csv_path = csv_path.resolve()
    assert csv_path.exists(), f"CSV 不存在：{csv_path}"

    # 读取时间列并推断 interval
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

    # 起止时间
    if cfg.get("start") and cfg.get("end"):
        start_dt = pd.to_datetime(cfg["start"])
        end_dt = pd.to_datetime(cfg["end"])
    else:
        start_dt = pd.to_datetime(dt_series.min())
        end_dt = pd.to_datetime(dt_series.max())

    # 交易参数
    rate = float(cfg.get("rate", 2.5 / 10000))
    slippage = float(cfg.get("slippage", 1))
    size = int(cfg.get("size", 10))
    pricetick = float(cfg.get("pricetick", 1))
    capital = float(cfg.get("capital", 1_000_000))

    # 策略参数（原 strategy 字段 +（如有）features 字段映射给 DoubleSyStrategy）
    params = dict(cfg.get("strategy", {}))

    # 若你的 CONFIG 里给了双鱼指标的特征参数（推荐）：
    # 例：
    # "features": {"algo": "shuangyu", "version": "1", "db": "", "interval": "1d"}
    feats = dict(cfg.get("features", {}))
    if feats:
        # DoubleSyStrategy 需要的参数名（可按你的策略类实际需要调整/删改）
        params.setdefault("algo_name", feats.get("algo", "shuangyu"))
        params.setdefault("feature_interval", feats.get("interval", cfg.get("interval", "1d")))
        params.setdefault("feature_version", feats.get("version", "1"))
        # 将 CONFIG 的 start/end 传给策略做 on_start 的取数窗口
        params.setdefault("feature_start", str(start_dt))
        params.setdefault("feature_end", str(end_dt))
        if feats.get("db"):
            params.setdefault("db_override_path", feats["db"])

    vt_symbol = f"{symbol}.{exchange.value}"
    out_dir = (OUTPUT_ROOT / vt_symbol.replace(".", "_")).resolve()

    return RunArgs(
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
        csv_path=csv_path,
        symbol=symbol,
        exchange=exchange,
    )