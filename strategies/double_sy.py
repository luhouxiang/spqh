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
import logging
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

    def calc_delta_position(self, price, row: SyRow, is_open_price: bool=True) -> int:
        """
        计算开盘后的仓位变化数:
        - 如果是开盘价来计算最多返回1
        - 如果是最低价来计算最多返回2

        返回: int
        """
        if np.isnan(row.sx1) or np.isnan(row.dnl1):
            return 0
        if is_open_price:
            if price < row.sx1 - row.dnl1:
                return 1
            else:
                return 0
        else:
            if price < row.sx1 - row.dnl1 - row.dnl1:
                return 2
            elif price < row.sx1 - row.dnl1:
                return 1
            else:
                return 0

    def calc_open_add_after_open(self, bar: BarData) -> int:
        """
        计算开盘后是否应新增 1 手仓位。

        规则：若昨日的 sx1 或 dnl1 缺失返回 0；若当日开盘价 < (sx1 - dnl1) 则返回 1，否则返回 0。

        参数：
            - bar: 当日的 BarData（用于获取开盘价及时间戳）

        返回：0 或 1
        """
        # 取到昨天的特征行用于判断
        row = self._get_prev_feats(bar.datetime)
        if row is None:
            return 0

        delta = 0
        try:
            delta = self.calc_delta_position(bar.open_price, row, True)
        except Exception:
            pass
        return delta
    
    def calc_low_add_after_lowest(self, bar: BarData) -> int:
        row = self._get_prev_feats(bar.datetime)
        if row is None:
            return 0

        delta = 0
        try:
            delta = self.calc_delta_position(bar.low_price, row, False)
        except Exception:
            pass
        return delta

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

    def _get_cur_position(self) -> int:
        """获取到今天的仓位
        """
        return self.pos

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
        row = self._get_prev_feats(ts)
        position = (self._get_prev_new_position(ts) + 1 + self.calc_init_position(bar.open_price, row)
                    + self._get_prev_pending_order(ts))
        return position if position < self.max_lots else self.max_lots

    def calc_open_expected_position(self, bar: BarData) -> int:
            """计算震荡日开盘时的期望持仓：
            昨天新建的多头仓位 + 1，并向上限 self.max_lots 截断。

            参数：
                - bar: 当日的 BarData（用于获取开盘价及时间戳）

            返回：int（期望持仓手数，>=0）。
            """
            # 昨天新建的仓位（只取新增的多头仓位数）
            yesterday_new = int(self._get_prev_new_position(bar.datetime))
            expected = yesterday_new + 1
            return int(expected)
        
    

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

    def add_position(self, count: int, price: float, is_long: bool):
        """根据给定的价格，增加对应的仓位"""
        if count <= 0:
            return 0
        capacity = self.max_lots - abs(self.pos)
        to_add = int(min(count, max(0, capacity)))
        if to_add <= 0:
            self.write_log("加仓请求被忽略：已达到最大手数上限")
            return 0
        # 下单（一次性按手数聚合成交量）
        if is_long:
            self.buy(price, volume=to_add)
            self.write_log(f"现持 {self.pos} 手, 当前加仓：价格 {price}，新增 {to_add} 手")
        else:
            self.short(price, volume=to_add)
            self.write_log(f"现持 {self.pos} 手, 当前加仓：价格 {price}，新增 {-to_add} 手")
        self.put_event()

    def sub_position(self, count: int, price: float, is_long: bool):
        """根据给定的价格，平掉对应的仓位"""
        if count <= 0:
            return 0
        to_close = int(min(count, abs(self.pos)))
        if to_close <= 0:
            self.write_log("减仓请求被忽略：当前无可平手数")
            return 0
        # 发单：多仓用 sell；若意外出现空仓则用 cover（兜底）
        if is_long:
            self.sell(price, volume=to_close)
        else:
            self.cover(price, volume=to_close)
        self.write_log(f"现持 {self.lots} 手, 减仓：价格 {price}，平掉 {to_close} 手")
        self.put_event()

    def _force_match_now(self, bar: BarData) -> None:
        """
        仅用于回测：在当前bar内立即撮合挂单（限价/本地止损单）。
        兼容不同版本的函数签名（带bar或不带bar）。
        """
        eng = getattr(self, "cta_engine", None)
        if eng is None:
            return

        def _call(func_name: str):
            fn = getattr(eng, func_name, None)
            if callable(fn):
                try:
                    fn(bar)  # 优先带 bar
                except TypeError:
                    try:
                        fn()  # 再试无参
                    except Exception:
                        pass

        _call("cross_limit_order")
        _call("cross_stop_order")

    # ---------------- 主体逻辑（只做多，日内最多加2手） ----------------
    def do_clear_long(self, bar: BarData):
        """平掉当前所有多头仓位（以当日开盘价），并立即撮合（回测需强制撮合）。"""
        # 仅在有净多仓时执行平多动作
        if self.pos > 0:
            # 使用已有的 clear_position 辅助方法进行平仓和日志记录
            self.clear_position(bar.open_price)
            # 在回测环境下尝试立刻撮合挂单/成交
            try:
                self._force_match_now(bar)
            except Exception:
                # 若引擎不支持强制撮合，仍保持平仓指令已下
                pass

    def do_open_long(self, bar: BarData):
        """简化：若为震荡日（self.dnl2_prev == 1），计算今日开盘应持仓：
        昨天新成交的多头仓位 + 1，且总体不超过 self.max_lots。

        返回计算得到的期望仓位（int），在非震荡日返回 None。
        """
        if self.dnl2_prev == 1:  # 如果是在震荡日，则重新计算预期仓位
            expected = self.calc_open_expected_position(bar)
        else:                   # 否则保持原有持仓+未完成订单
            expected = self.pos + self._get_prev_pending_order(bar.datetime)
            # 记录日志并返回
        label_1 = "普通日"
        if self.dnl2_prev == 1:
            label_1 = "震荡日"

        dlt = self.calc_delta_position(bar.open_price, self._get_prev_feats(bar.datetime), True)
        expected_position = min(self.max_lots, dlt + expected + self._get_prev_new_position(bar.datetime))
        real_position = self.pos
        if expected_position * real_position < 0:  # 期望仓位和实际仓位相反，先清仓，再走期望仓位
            logging.info(
                f"[{self.am.datetime_array[-1]}]{label_1}开盘前期望持仓：{expected}手，开盘时期望持仓:{expected_position}手，实际仓位{real_position}手")
            # 先清仓，再建仓
            self.clear_position(bar.open_price)
            self._force_match_now(bar)
            self.add_position(expected_position, bar.open_price, expected_position > 0)
            self._force_match_now(bar)
        else:
            add_pos = expected_position - real_position
            if add_pos != 0:
                logging.info(f"[{self.am.datetime_array[-1]}]{label_1}开盘前期望持仓：{expected}手，开盘时期望持仓:[{self.pos}]/[{expected_position}]手")
            if expected_position - real_position > 0:
                self.add_position(add_pos, bar.open_price, add_pos > 0)
                self._force_match_now(bar)
            elif expected_position - real_position < 0:
                self.sub_position(abs(add_pos), bar.open_price, add_pos > 0)
                self._force_match_now(bar)
                
        return expected

    def on_bar(self, bar: BarData):
        self.am.update_bar(bar)
        if not self.am.inited:
            return
    
        f_prev = self._get_prev_feats(bar.datetime)  # 找不到之前的时间点，也退出
        if f_prev is None:
            self.put_event()
            return
        self.qsx2_prev = int(f_prev.qsx2) if not np.isnan(f_prev.qsx2) else 0
        self.dnl2_prev = int(f_prev.dnl2) if not np.isnan(f_prev.dnl2) else 0
        self.sx1_prev = int(f_prev.sx1) if not np.isnan(f_prev.sx1) else 0
        if self.qsx2_prev == 1:  # qsx2为1时开始做多
            self.do_open_long(bar)
        else:
            self.do_clear_long(bar)    # 为0时平仓停止做多
            
            

    # def on_bar(self, bar: BarData):
    #     self.am.update_bar(bar)
    #     if not self.am.inited:
    #         return

    #     f_prev = self._get_prev_feats(bar.datetime)
    #     if f_prev is None:
    #         self.put_event()
    #         return
    #     """
    #     1、qsx2_prev为1时做多， 否则平多
    #     2、dnl2_prev为震荡区间，表达的为重新开始。保留昨天加仓数+1
    #     3、开盘价低于第一个加仓台阶，以开盘价加仓1次
    #     4、最低价低于加仓台阶，且没有加满，则加仓1次或2次
    #     5、来不及加的仓位记入第二天待加仓
    #     """
    #     self.qsx2_prev = int(f_prev.qsx2) if not np.isnan(f_prev.qsx2) else 0
    #     self.dnl2_prev = int(f_prev.dnl2) if not np.isnan(f_prev.dnl2) else 0
    #     self.sx1_prev = int(f_prev.sx1) if not np.isnan(f_prev.sx1) else 0
    #     if self.qsx2_prev == 1:  # 为1时做多，否则平多
    #         if self.dnl2_prev == 1:  # 区间标志
    #             # 仓位调整
    #             expected_position = self.calc_today_position(bar, bar.datetime)
    #             real_position = self._get_cur_position(bar.datetime)
    #             if expected_position * real_position < 0:  # 期望仓位和实际仓位相反，只认实际仓位
    #                 # 先清仓，再建仓
    #                 self.clear_position(bar.open_price)
    #                 self._force_match_now(bar)
    #                 self.add_position(expected_position, bar.open_price, expected_position > 0)
    #                 self._force_match_now(bar)
    #             else:
    #                 if abs(expected_position) - abs(real_position) > 0:
    #                     self.add_position(expected_position, bar.open_price, expected_position > 0)
    #                     self._force_match_now(bar)
    #                 elif abs(expected_position) - abs(real_position) < 0:
    #                     self.sub_position(expected_position, bar.open_price, expected_position > 0)
    #                     self._force_match_now(bar)
    #         else:
    #             pass
    #     else:
    #         self.clear_position(bar.open_price)
    #         self._force_match_now(bar)

    #     self.put_event()
    #     return



