from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from common.datetime_normalize import normalize_datetime_columns

from vnpy_ctastrategy import CtaTemplate
from vnpy.trader.object import BarData
from vnpy.trader.utility import ArrayManager
from vnpy.trader.constant import Interval, Direction, Offset

import logging
# 双鱼特征字段全集（库内列名）
SY_COLS = ["lj","qs1","dnl1","qsx1","sx1","qs2","dnl2","qsx2","sx2","phqd","lsqd"]


@dataclass
class SyRow:
    ts: pd.Timestamp
    lj: float = np.nan
    qs1: float = np.nan
    dnl1: float = np.nan
    qsx1: float = np.nan
    sx1: float = np.nan
    qs2: float = np.nan
    dnl2: float = np.nan
    qsx2: float = np.nan
    sx2: float = np.nan
    phqd: float = np.nan
    lsqd: float = np.nan


def _to_naive_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # tz-aware -> 去时区；naive -> 原样返回
    if getattr(idx, "tz", None):
        i = idx.tz_convert(None)
    else:
        i = idx
    return i
    # return idx.tz_convert(None) if getattr(idx, "tz", None) is not None else idx


def _to_naive_ts(ts) -> pd.Timestamp:
    # tzinfo -> 去时区：naive -> 原样返回
    t = pd.Timestamp(ts)

    if getattr(t, "tzinfo", None):
        n = t.tz_convert(None)
    else:
        n = t
    return n
    # return t.tz_convert(None) if getattr(t, "tzinfo", None) is not None else t


class LoggingCtaTemplate(CtaTemplate):
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        pass

        # 重写：拦截并记日志，再调用父类的 send_order

    def send_order(
            self,
            direction: Direction,
            offset: Offset,
            price: float,
            volume: float,
            stop: bool = False,
            lock: bool = False,
            net: bool = False,
    ) -> list[str]:
        # 调用父类真正下单
        orderids = super().send_order(direction, offset, price, volume, stop, lock, net)
        logging.info(f"[{self.am.datetime_array[-1]}]send_order【{orderids}】: {direction}, {offset}, {volume}, {price}")
        return orderids


class DoubleSyStrategy(LoggingCtaTemplate):
    author= "luhx"
    
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        from vnpy.trader.database import get_database
        self.am = ArrayManager(1)
        self.db_override_path = get_database().db.database
        self.feature_start = None
        self.feature_end = None
        self.feature_interval = None
        self.feature_version = "1"
        self.max_lots: int = 5  # 总仓位上限（1~5）
        self.algo_name: str = "shuangyu"  # 特征表算法名
        self.exchange = None
        self.order_count = 0    # 订单数

        self.qsx2_prev = 0
        self.dnl2_prev = 0


    def on_init(self):
        self.write_log("策略初始化")
        self.load_bar(50)

    def on_start(self):
        """在这里从数据库加载当前品种的双鱼特征"""
        self.write_log("策略启动：开始加载双鱼特征")
        try:
            from common.algo_features_store import load_algo_features
            from common.args_from_config import compute_run_args_from_config
            symbol, exchange = self.vt_symbol.split(".")
            override = None if not self.db_override_path else self.db_override_path
            args = compute_run_args_from_config(symbol)
            self.feature_start = args.start_dt
            self.feature_end = args.end_dt
            self.feature_interval = args.interval.value
            self.exchange = args.exchange.value
            df = load_algo_features(
                algo_name=self.algo_name,
                symbol=symbol,
                exchange=exchange,
                interval=self.feature_interval,
                start_dt=self.feature_start,
                end_dt=self.feature_end,
                feature_cols=SY_COLS,
                version=self.feature_version,
                override_path=override,
            )
            if df is None or df.empty:
                self.write_log("双鱼特征为空（检查入库/时间窗口/interval/version）")
            else:
                self._prepare_features(df)
                self.write_log(f"双鱼特征加载完成：{len(self._feat_df)} 条")

        except Exception as e:
            self.write_log(f"加载双鱼特征失败：{e}")

    def _prepare_features(self, df: pd.DataFrame) -> None:
        """
        将原始双鱼特征标准化为：
          - 索引：tz-naive 的 DatetimeIndex（升序，去重）
          - 列：严格为 SY_COLS（缺失补 NaN，按既定顺序）
        """
        # 1) 复制与列名标准化
        x = df.copy()
        x.columns = [c.lower().strip() for c in x.columns]

        # 2) 确保时间索引
        if not isinstance(x.index, pd.DatetimeIndex):
            if "datetime" not in x.columns:
                raise ValueError("algo_features 需 index=DatetimeIndex 或包含 'datetime' 列")
            # 解析时间；坏值剔除
            x["datetime"] = pd.to_datetime(x["datetime"], errors="coerce")
            x = x.dropna(subset=["datetime"]).set_index("datetime")

        # 3) 统一为 tz-naive，去重并排序
        x.index = _to_naive_index(x.index)
        # 对重复时间保留最后一条，避免冲突
        x = x[~x.index.duplicated(keep="last")].sort_index()

        # 4) 只保留需要的特征列（缺失补 NaN；列顺序固定）
        x = x.reindex(columns=SY_COLS, fill_value=np.nan)

        # 5) 缓存
        self._feat_df = x
        self._feat_idx = x.index

    # ---------------- 主体逻辑（只做多，日内最多加2手） ----------------
    def on_bar(self, bar: BarData):
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        f_prev = self._get_prev_feats(_to_naive_ts(bar.datetime))
        if f_prev is None:
            self.put_event()
            return

        self.qsx2_prev = int(f_prev.qsx2) if not np.isnan(f_prev.qsx2) else 0
        self.dnl2_prev = int(f_prev.dnl2) if not np.isnan(f_prev.dnl2) else 0

        if self.dnl2_prev == 1:
            # 仓位调整
            pass
        else:
            pass


    def _get_prev_feats(self, ts: pd.Timestamp) -> SyRow:
        """
        返回 (F_{t-1}, prev_ts)：
        - prev_ts 为该行特征对应的时间，用于“昨建”判断（账本保留）
        """
        if self._feat_df is None or self._feat_idx is None or len(self._feat_idx) == 0:
            return None
        pos = self._feat_idx.searchsorted(ts, side="right") - 1
        if pos < 0:
            return None
        s = self._feat_df.iloc[pos]
        prev_ts = self._feat_idx[pos]

        def v(name):
            val = s.get(name, np.nan)
            return float(val) if pd.notna(val) else np.nan

        row = SyRow(ts=prev_ts,
            lj=v("lj"), qs1=v("qs1"), dnl1=v("dnl1"), qsx1=v("qsx1"), sx1=v("sx1"),
            qs2=v("qs2"), dnl2=v("dnl2"), qsx2=v("qsx2"), sx2=v("sx2"),
            phqd=v("phqd"), lsqd=v("lsqd"),
        )
        return row