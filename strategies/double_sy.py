# double_sy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from vnpy_ctastrategy import CtaTemplate
from vnpy.trader.object import BarData
from vnpy.trader.utility import ArrayManager
from vnpy.trader.constant import Interval, Direction, Offset

# 双鱼特征字段全集（库内列名）
SY_COLS = ["lj","qs1","dnl1","qsx1","sx1","qs2","dnl2","qsx2","sx2","phqd","lsqd"]


# ===================== 时区工具（统一到本地时区；不转为 naive） =====================
def _ensure_tz_index(idx: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """把索引统一成 tz-aware 且处于 tz 时区。"""
    if getattr(idx, "tz", None) is None:
        return idx.tz_localize(tz)
    return idx.tz_convert(tz)


def _ensure_tz_ts(ts, tz: str) -> pd.Timestamp:
    """把单个时间统一成 tz-aware 且处于 tz 时区。"""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(tz)
    return t.tz_convert(tz)


# ===================== 数据结构 =====================
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


# ===================== 策略实现（只做多，日内最多加2手） =====================
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

        self.qsx2_prev = 0  # 做多标志
        self.dnl2_prev = 0  # 昨日标志区间
        self.sx1_prev = 0   # 昨日下轨
        self._feat_df = None    # 双鱼指标
        self._feat_idx = None
        self.feature_timezone: str = "Asia/Shanghai"
        self._pending_order_df = None        # 订单 每天未完成的订单数  column 中含有datetime,order
        self._position_df = None     # 每天的持仓数   column 中含有datetime,position


    def on_init(self):
        self.write_log("策略初始化")
        self.load_bar(50)

    def on_start(self):
        """在这里从数据库加载当前品种的双鱼特征（统一到本地时区）"""
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
                self._prepare_features(df)  # 统一时区、去重、排序、保列
                self.write_log(f"双鱼特征加载完成：{len(self._feat_df)} 条")
        except Exception as e:
            self.write_log(f"加载双鱼特征失败：{e}")

    def on_stop(self):
        self.write_log("策略停止")

    # ---------------- 特征准备与访问 ----------------
    def _prepare_features(self, df: pd.DataFrame) -> None:
        """
        标准化双鱼特征：
          - 索引：tz-aware 的 DatetimeIndex（统一到 self.feature_timezone）
          - 列：严格为 SY_COLS（缺失补 NaN，按既定顺序）
        """
        x = df.copy()
        # 确保时间索引
        if not isinstance(x.index, pd.DatetimeIndex):
            if "datetime" not in x.columns:
                raise ValueError("algo_features 需 index=DatetimeIndex 或包含 'datetime' 列")
            x["datetime"] = pd.to_datetime(x["datetime"], errors="coerce")
            x = x.dropna(subset=["datetime"]).set_index("datetime")

        # 统一到本地时区（不做 tz-naive 转换）
        x.index = _ensure_tz_index(x.index, self.feature_timezone)

        # 去重、排序
        x = x[~x.index.duplicated(keep="last")].sort_index()

        # 只保留需要列
        x = x.reindex(columns=SY_COLS, fill_value=np.nan)

        # 缓存
        self._feat_df = x
        self._feat_idx = x.index



    def _get_prev_feats(self, ts: pd.Timestamp) -> SyRow:
        """
        返回 (F_{t-1}, prev_ts)：
        - prev_ts 为该行特征对应的时间，用于“昨建”判断（账本保留）
        - 统一把传入 ts 转为 tz-aware 且处于 feature_timezone
        """
        if self._feat_df is None or self._feat_idx is None or len(self._feat_idx) == 0:
            return None

        ts = _ensure_tz_ts(ts, self.feature_timezone)

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

    def calc_init_position(self, bar: BarData, row: SyRow):
        """计算的初始加仓数,只有开盘价小于昨天的下轨才能加一个仓位，否则初始加仓数为0"""
        if np.isnan(row.sx1) or np.isnan(row.dnl1):
            return 0
        if bar.open_price < row.sx1 - row.dnl1:
            return 1
        return 0

    def _get_prev_new_position(self, ts: pd.Timestamp) -> int:
        """获取昨天新增的仓位
        获取到昨天，前天的仓位，用昨天的仓位-前天的仓位，大于0时返回差值，否则返回0
        """
        if self._position_df is None or len(self._position_df) == 0:
            return 0

        ts = _ensure_tz_ts(ts, self.feature_timezone)

        pos = self._position_df.searchsorted(ts, side="right") - 1
        if pos < 0:
            return 0
        if pos - 1 < 0:  # 前天没有仓位，昨天的仓位即为新增仓位
            return self._position_df[pos]

        if self._position_df[pos]["position"] * self._position_df[pos-1]["position"] < 0:   # 两天持仓相反
            return 0

        dtl = abs(self._position_df[pos]["position"]) - abs(self._position_df[pos-1]["position"])
        if dtl > 0:
            return dtl
        else:
            return 0

    def _get_cur_position(self, ts: pd.Timestamp) -> int:
        """获取到今天的仓位
        """
        if self._position_df is None or len(self._position_df) == 0:
            return 0

        ts = _ensure_tz_ts(ts, self.feature_timezone)

        pos = self._position_df.searchsorted(ts, side="right")
        if pos < 0:
            return 0
        return self._position_df[pos]["position"]

    def _get_prev_pending_order(self, ts: pd.Timestamp) -> int:
        """获取昨天未完成的订单（返回0表示没有，负数表示做空数，正数表示做多数，目的是在开盘时用开盘价兑换
        平多表示做多，平空表示做多
        """
        if self._pending_order_df is None or len(self._pending_order_df) == 0:
            return 0

        ts = _ensure_tz_ts(ts, self.feature_timezone)

        pos = self._pending_order_df.searchsorted(ts, side="right") - 1
        if pos < 0:
            return 0
        return self._pending_order_df[pos]["order"]

    def calc_today_position(self, bar: BarData, ts: pd.Timestamp):
        """计算今天应有仓位
        今天应有仓位 = 昨天新建仓位+1手+计算的初始加仓数+昨天未完成订单
        大于5手返回5，否则返回今天应有仓位
        """
        position = self._get_prev_new_position(ts) + 1 + self.calc_init_position(bar) + self._get_prev_pending_order(ts)
        return position if position < self.max_lots else self.max_lots

    def clear_position(self, price: float):
        """根据给定的价格，清除所有的仓位"""
        lots_before = self.pos

        # 先发单（根据净持仓方向）
        if self.pos > 0:
            self.sell(price, volume=abs(self.pos))
        elif self.pos < 0:
            self.cover(price, volume=abs(self.pos))

        closed_lots = self.pos
        self.write_log(f"清仓：价格 {price}，平掉 {closed_lots} 手（原 {lots_before} 手）")
        self.put_event()    # 这儿动作过后，考虑自动撮合 TODO: 稍后测试这一块

    def add_position(self, bar: BarData, count: int, price: float, longer: bool):
        """根据给定的价格，增加对应的仓位"""
        if count <= 0:
            return 0
        capacity = self.max_lots - abs(self.pos)
        to_add = int(min(count, max(0, capacity)))
        if to_add <= 0:
            self.write_log("加仓请求被忽略：已达到最大手数上限")
            return 0
        # 下单（一次性按手数聚合成交量）
        if longer:
            self.buy(price, volume=to_add)
            self.write_log(f"现持 {self.pos} 手, 当前加仓：价格 {price}，新增 {to_add} 手")
        else:
            self.short(price, volume=to_add)
            self.write_log(f"现持 {self.pos} 手, 当前加仓：价格 {price}，新增 {-to_add} 手")
        self.put_event()

    def sub_position(self, bar: BarData, count: int, price: float):
        """根据给定的价格，平掉对应的仓位"""
        if count <= 0:
            return 0
        to_close = int(min(count, abs(self.pos)))
        if to_close <= 0:
            self.write_log("减仓请求被忽略：当前无可平手数")
            return 0
        # 发单：多仓用 sell；若意外出现空仓则用 cover（兜底）
        if self.pos >= 0:
            self.sell(price, volume=to_close)
        else:
            self.cover(price, volume=to_close)
        self.write_log(f"现持 {self.lots} 手, 减仓：价格 {price}，平掉 {to_close} 手")
        self.put_event()

    # ---------------- 主体逻辑（只做多，日内最多加2手） ----------------
    def on_bar(self, bar: BarData):
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        f_prev = self._get_prev_feats(bar.datetime)
        if f_prev is None:
            self.put_event()
            return

        self.qsx2_prev = int(f_prev.qsx2) if not np.isnan(f_prev.qsx2) else 0
        self.dnl2_prev = int(f_prev.dnl2) if not np.isnan(f_prev.dnl2) else 0
        self.sx1_prev = int(f_prev.sx1) if not np.isnan(f_prev.sx1) else 0

        if self.dnl2_prev == 1:
            # 仓位调整
            expected_position = self.calc_today_position(bar, bar.datetime)
            real_position = self._get_cur_position(bar.datetime)
            if expected_position * real_position < 0:   # 期望仓位和实际仓位相反，只认实际仓位
                # 先清仓，再建仓
                pass
            else:
                if abs(expected_position) - abs(real_position) > 0:
                    # 加仓
                    pass
                elif abs(expected_position) - abs(real_position) < 0:
                    # 减仓
                    pass
        else:
            pass

        self.put_event()
        return



