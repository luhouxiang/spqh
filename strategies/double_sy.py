# double_sy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from vnpy_ctastrategy import CtaTemplate
from vnpy.trader.object import BarData
from vnpy.trader.utility import ArrayManager

# 双鱼特征字段全集（库内列名）
SY_COLS = ["lj","qs1","dnl1","qsx1","sx1","qs2","dnl2","qsx2","sx2","phqd","lsqd"]


@dataclass
class SyRow:
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
    return idx.tz_convert(None) if getattr(idx, "tz", None) is not None else idx

def _to_naive_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert(None) if getattr(t, "tzinfo", None) is not None else t


class DoubleSyStrategy(CtaTemplate):
    """
    双鱼策略（只做多；on_start 从数据库加载特征；on_bar 用“昨日指标”交易）

    规则要点（与你最新描述一致）：
    - 昨日 qsx2=1 → 做多模式；昨日 qsx2=0 → 今日开盘清仓并停止做多
    - 仓位单位为“手”（fixed_size 决定 1 手对应的下单量），总上限 max_lots（默认 5 手）
    - 区间重置（昨日 dnl2=1）：今日开盘平掉“昨日前”的仓位，保留昨建，再以开盘价新建 1 手
    - 加仓（仅当做多，且启用 lj 过滤时 lj≥阈值）：
        以昨日 sx1、dnl1 形成两级台阶：L1 = sx1 - dnl1；L2 = sx1 - 2*dnl1
        当日开盘 ≤ L1：先用开盘加 1 手
        当日最低 ≤ L2：当日总计加到 2 手（若开盘没加过，则 L1、L2 各加 1 手；若开盘加过，则再在 L2 加 1 手）
        否则若最低 ≤ L1：当日总计加 1 手（限价 L1；若开盘已加过则不再加）
    - “昨天所建仓位今天不平”（除非昨日 dnl2=1 才触发区间重置）

    数据来源：
    - on_start() 通过 algo_features_store.load_algo_features(...) 从数据库加载当前品种特征（使用策略参数）
    """

    author = "you"

    # === 交易参数 ===
    fixed_size: int = 1            # 1 手对应的下单数量
    max_lots: int = 5              # 总仓位上限（1~5）
    use_lj_filter: bool = True     # 是否启用 lj 过滤
    lj_threshold: float = 7.0      # lj 门槛（≥此值才允许开/加）

    # === 特征加载参数（由 CONFIG 为每个品种传入） ===
    algo_name: str = "shuangyu"    # 特征表算法名
    feature_interval: str = "1d"   # 入库时用于区分间隔的字符串（与写表一致，如 "1d"/"1m"）
    feature_version: str = "1"     # 特征版本
    feature_start: str = "1970-01-01 00:00:00"  # 加载起始时间（可用该品种 CONFIG 的 start）
    feature_end: str   = "2100-01-01 00:00:00"  # 加载结束时间（可用该品种 CONFIG 的 end）
    db_override_path: str = ""     # 可选：强制指定 sqlite 文件路径（留空=自动探测/回退）

    parameters = [
        # 交易
        "fixed_size", "max_lots", "use_lj_filter", "lj_threshold",
        # 特征加载
        "algo_name", "feature_interval", "feature_version",
        "feature_start", "feature_end", "db_override_path",
    ]
    variables = ["lots", "mode", "last_qsx2", "last_dnl2"]

    # 运行时变量
    lots: int = 0                  # 当前总手数（0~max_lots）
    mode: str = "flat"             # "flat"|"long"
    last_qsx2: int = 0             # 昨日 qsx2（便于观察）
    last_dnl2: int = 0             # 昨日 dnl2（便于观察）

    # 内部缓存：特征表（index=DatetimeIndex）与“分桶账本”（记录每日新增手数）
    _feat_df: Optional[pd.DataFrame] = None
    _feat_idx: Optional[pd.DatetimeIndex] = None
    _unit_ledger: List[Tuple[pd.Timestamp, int]]  # [(open_ts, lots_that_day), ...]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.am = ArrayManager(1)
        self._unit_ledger = []
        self.db_override_path = None

    # ---------------- 生命周期 ----------------
    def on_init(self):
        self.write_log("策略初始化")
        self.load_bar(50)
        from vnpy.trader.database import get_database
        db = get_database()
        self.db_override_path = db.db.database

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

    def on_stop(self):
        self.write_log("策略停止")

    # ---------------- 特征准备与访问 ----------------
    import pandas as pd


    def _prepare_features(self, df: pd.DataFrame):
        x = df.copy()
        x.columns = [c.lower().strip() for c in x.columns]
        for c in SY_COLS:
            if c not in x.columns:
                x[c] = np.nan
        if not isinstance(x.index, pd.DatetimeIndex):
            if "datetime" in x.columns:
                x["datetime"] = pd.to_datetime(x["datetime"])
                x = x.set_index("datetime")
            else:
                raise ValueError("algo_features 需 index=DatetimeIndex 或包含 'datetime' 列")
        x = x.sort_index()
        self._feat_df = x[SY_COLS]
        self._feat_df.index = _to_naive_index(self._feat_df.index)
        self._feat_idx = self._feat_df.index

    def _get_prev_feats(self, ts: pd.Timestamp) -> Tuple[Optional[SyRow], Optional[pd.Timestamp]]:
        """
        返回 (F_{t-1}, prev_ts)：
        - prev_ts 为该行特征对应的时间，用于“昨建”判断（账本保留）
        """
        if self._feat_df is None or self._feat_idx is None or len(self._feat_idx) == 0:
            return None, None
        ts = _to_naive_ts(ts)  # ★ 关键：把传入时间也变为 naive
        pos = self._feat_idx.searchsorted(pd.Timestamp(ts), side="right") - 1
        if pos < 0:
            return None, None
        s = self._feat_df.iloc[pos]
        prev_ts = self._feat_idx[pos]

        def v(name):
            val = s.get(name, np.nan)
            return float(val) if pd.notna(val) else np.nan

        row = SyRow(
            lj=v("lj"), qs1=v("qs1"), dnl1=v("dnl1"), qsx1=v("qsx1"), sx1=v("sx1"),
            qs2=v("qs2"), dnl2=v("dnl2"), qsx2=v("qsx2"), sx2=v("sx2"),
            phqd=v("phqd"), lsqd=v("lsqd"),
        )
        return row, prev_ts

    # ---------------- 账本与下单辅助 ----------------
    def _base_lot(self) -> int:
        return max(int(self.fixed_size), 1)

    def _max_lots(self) -> int:
        return max(1, min(int(self.max_lots), 5))

    def _ledger_add(self, open_ts: pd.Timestamp, lots_to_add: int):
        if lots_to_add <= 0:
            return
        if self._unit_ledger and self._unit_ledger[-1][0] == open_ts:
            ts, old = self._unit_ledger[-1]
            self._unit_ledger[-1] = (ts, old + lots_to_add)
        else:
            self._unit_ledger.append((open_ts, lots_to_add))
        self.lots += lots_to_add

    def _ledger_close_older_than(self, keep_ts: pd.Timestamp) -> int:
        """平掉开仓时间 < keep_ts 的所有手数（保留昨建）"""
        to_keep: List[Tuple[pd.Timestamp, int]] = []
        to_close = 0
        for ts, n in self._unit_ledger:
            if ts < keep_ts:
                to_close += n
            else:
                to_keep.append((ts, n))
        self._unit_ledger = to_keep
        self.lots -= to_close
        return to_close

    def _flatten_all(self, price: float):
        if self.pos > 0:
            self.sell(price, volume=abs(self.pos))
        self._unit_ledger = []
        self.lots = 0
        self.mode = "flat"

    # ---------------- 主体逻辑（只做多，日内最多加2手） ----------------
    def on_bar(self, bar: BarData):
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        f_prev, prev_ts = self._get_prev_feats(bar.datetime)
        if f_prev is None:
            self.put_event()
            return

        qsx2_prev = int(f_prev.qsx2) if not np.isnan(f_prev.qsx2) else 0
        dnl2_prev = int(f_prev.dnl2) if not np.isnan(f_prev.dnl2) else 0
        self.last_qsx2 = qsx2_prev
        self.last_dnl2 = dnl2_prev

        def pass_lj() -> bool:
            return (not self.use_lj_filter) or (pd.notna(f_prev.lj) and f_prev.lj >= float(self.lj_threshold))

        max_lots = self._max_lots()
        lot_unit = self._base_lot()

        # 0) 停止：昨日 qsx2=0 → 今日开盘清仓并停止
        if qsx2_prev == 0:
            if self.pos > 0:
                self.sell(bar.open_price, volume=abs(self.pos))
            self._unit_ledger = []
            self.lots = 0
            self.mode = "flat"
            self.put_event()
            return

        # 1) 做多模式（qsx2_prev == 1）
        # 1.1 区间重置：昨日 dnl2==1
        if dnl2_prev == 1:
            if prev_ts is not None and self.lots > 0:
                close_lots = self._ledger_close_older_than(prev_ts)
                if close_lots > 0:
                    self.sell(bar.open_price, volume=close_lots * lot_unit)
                    self.write_log(f"区间重置：平掉更早仓位 {close_lots} 手 @{bar.open_price}")
            if pass_lj():
                self.buy(bar.open_price, volume=lot_unit)
                self._ledger_add(bar.datetime, 1)
                self.mode = "long"
                self.write_log(f"区间开启：开盘建 1 手，现持 {self.lots} 手")

        # 1.2 当日加仓（仅两级台阶 L1/L2；当日最多加2手；总仓位不超过 max_lots）
        if pass_lj() and pd.notna(f_prev.sx1) and pd.notna(f_prev.dnl1) and f_prev.dnl1 > 0:
            L1 = f_prev.sx1 - 1.0 * f_prev.dnl1
            L2 = f_prev.sx1 - 2.0 * f_prev.dnl1

            daily_target_add = 0
            open_add = False

            # 开盘触发 L1
            if bar.open_price <= L1:
                daily_target_add = 1
                open_add = True

            # 盘中触发
            if bar.low_price <= L2:
                daily_target_add = 2
            elif bar.low_price <= L1:
                daily_target_add = max(daily_target_add, 1)

            capacity = max_lots - self.lots
            to_add = min(daily_target_add, max(0, capacity))

            if to_add > 0:
                # 先用开盘加（若命中条件且仍有容量）
                if open_add and to_add > 0:
                    self.buy(bar.open_price, volume=lot_unit)
                    self._ledger_add(bar.datetime, 1)
                    to_add -= 1
                    self.write_log(f"开盘加仓 1 手，现持 {self.lots} 手")

                # 剩余用限价挂台阶
                if to_add > 0:
                    if bar.low_price <= L2:
                        # 目标两手：若开盘没加过 -> L1,L2 各1；若已加过 -> 再在 L2 1手
                        if not open_add and to_add > 0:
                            self.buy(L1, volume=lot_unit)
                            self._ledger_add(bar.datetime, 1)
                            to_add -= 1
                            self.write_log(f"限价加仓 1 手@L1={L1:.4f}，现持 {self.lots} 手")
                        if to_add > 0:
                            self.buy(L2, volume=lot_unit)
                            self._ledger_add(bar.datetime, 1)
                            to_add -= 1
                            self.write_log(f"限价加仓 1 手@L2={L2:.4f}，现持 {self.lots} 手")
                    elif bar.low_price <= L1:
                        if not open_add and to_add > 0:
                            self.buy(L1, volume=lot_unit)
                            self._ledger_add(bar.datetime, 1)
                            to_add -= 1
                            self.write_log(f"限价加仓 1 手@L1={L1:.4f}，现持 {self.lots} 手")

        self.put_event()
