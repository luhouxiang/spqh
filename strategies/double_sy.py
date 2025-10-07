# double_sy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from vnpy_ctastrategy import CtaTemplate
from vnpy.trader.object import BarData
from vnpy.trader.utility import ArrayManager


@dataclass
class SyRow:
    """双鱼特征的一行数据（用于快速访问）"""
    lj: float
    qs1: float
    dnl1: float
    qsx1: float
    sx1: float
    qs2: float
    dnl2: float
    qsx2: float
    sx2: float
    phqd: float
    lsqd: float


class DoubleSyStrategy(CtaTemplate):
    """
    双鱼策略（多品种可用）
    规则要点：
      - 用“昨日信号”决定今天的开/平与区间切换；
      - 加仓台阶用“昨日 sx1/dnl1”（做多）或“昨日 qsx1/dnl1”（做空）；
      - 最多5倍（含初始1倍），开盘越过首台阶则按开盘价加第一倍，其余台阶用限价；
      - 可选 lj 过滤（默认门槛 7）。
    """

    author = "you"

    # ===== 参数 =====
    fixed_size: int = 1            # 1倍对应的手数
    max_multiplier: int = 5        # 最大倍数（1~5）
    lj_threshold: float = 7.0      # 仅当lj>=此值才允许开/加（为0则不启用）
    use_lj_filter: bool = True     # 是否启用lj过滤
    algo_name: str = "shuangyu"    # 特征表算法名
    interval_text: str = "1d"      # 入库时的interval文本（需与你写表一致：如"1d"/"1m"）
    auto_load_features: bool = True  # 若未注入df，是否自动从数据库加载
    # 允许通过 setting 注入：algo_features=pd.DataFrame（index=datetime），列包含双鱼字段

    # ===== 变量（监控用）=====
    units: int = 0                 # 当前倍数（0~5）
    mode: str = "flat"             # "flat"|"long"|"short"
    last_qsx2: int = 0             # 昨日long标志缓存
    last_sx2: int = 0              # 昨日short标志缓存

    parameters = [
        "fixed_size", "max_multiplier",
        "lj_threshold", "use_lj_filter",
        "algo_name", "interval_text", "auto_load_features"
    ]
    variables = ["units", "mode", "last_qsx2", "last_sx2"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.am = ArrayManager(100)

        # 接收或自动加载特征表
        self.features_df: Optional[pd.DataFrame] = setting.get("algo_features")
        self._feat_map: Optional[pd.DataFrame] = None  # 按时间排序的DF（index=DatetimeIndex）
        self._feat_index: Optional[pd.DatetimeIndex] = None

        if self.features_df is not None:
            self._prepare_features(self.features_df)
        elif self.auto_load_features:
            self._try_auto_load_features()
        else:
            self.write_log("未注入双鱼特征，且关闭自动加载；仅做占位。")

    # ========== 初始化与状态 ==========
    def on_init(self):
        self.write_log("策略初始化")
        # 预加载一定历史bar用于指标/边界检查
        self.load_bar(50)

    def on_start(self):
        self.write_log("策略启动")

    def on_stop(self):
        self.write_log("策略停止")

    # ========== 特征载入 ==========
    def _prepare_features(self, df: pd.DataFrame):
        """确保 index=DatetimeIndex 且列名标准化"""
        if df is None or len(df) == 0:
            self._feat_map = None
            self._feat_index = None
            self.write_log("双鱼特征为空")
            return
        x = df.copy()
        x.columns = [c.lower().strip() for c in x.columns]
        # 只保留需要的列
        cols = ["lj","qs1","dnl1","qsx1","sx1","qs2","dnl2","qsx2","sx2","phqd","lsqd"]
        miss = [c for c in cols if c not in x.columns]
        for c in miss:
            x[c] = np.nan
        # index
        if not isinstance(x.index, pd.DatetimeIndex):
            if "datetime" in x.columns:
                x["datetime"] = pd.to_datetime(x["datetime"])
                x = x.set_index("datetime")
            else:
                raise ValueError("algo_features 需要 index=DatetimeIndex 或包含 'datetime' 列")
        x = x.sort_index()
        self._feat_map = x[cols]
        self._feat_index = x.index
        self.write_log(f"已载入双鱼特征 {len(self._feat_map)} 条")

    def _try_auto_load_features(self):
        """无注入时，尝试从sqlite/同库自动读取该品种的全部双鱼特征"""
        try:
            from common.algo_features_store import load_algo_features  # 你之前放的工具模块
            symbol, exchange = self.vt_symbol.split(".")
            # 宽范围读取（实际是全量）；interval需与入库一致
            df = load_algo_features(
                algo_name=self.algo_name,
                symbol=symbol,
                exchange=exchange,
                interval=self.interval_text,
                start_dt="1970-01-01 00:00:00",
                end_dt="2100-01-01 00:00:00",
                feature_cols=["lj","qs1","dnl1","qsx1","sx1","qs2","dnl2","qsx2","sx2","phqd","lsqd"],
                version="1",
            )
            self._prepare_features(df)
        except Exception as e:
            self.write_log(f"自动加载双鱼特征失败：{e}")

    # ========== 工具：取昨日/前日特征 ==========
    def _get_feat_prev_and_prevprev(self, ts) -> Tuple[Optional[SyRow], Optional[SyRow]]:
        """返回 (F_{t-1}, F_{t-2})；用 index.asof 逻辑：取 <= ts 的最近记录与其前一条"""
        if self._feat_map is None or self._feat_index is None or len(self._feat_index) == 0:
            return None, None
        # 找到 <= ts 的位置
        pos = self._feat_index.searchsorted(pd.Timestamp(ts), side="right") - 1
        if pos < 0:
            return None, None
        prev = self._feat_map.iloc[pos]
        prevprev = self._feat_map.iloc[pos - 1] if pos - 1 >= 0 else None

        def to_row(s) -> SyRow:
            return SyRow(
                lj=float(s.get("lj", np.nan)) if pd.notna(s.get("lj")) else np.nan,
                qs1=float(s.get("qs1", np.nan)) if pd.notna(s.get("qs1")) else np.nan,
                dnl1=float(s.get("dnl1", np.nan)) if pd.notna(s.get("dnl1")) else np.nan,
                qsx1=float(s.get("qsx1", np.nan)) if pd.notna(s.get("qsx1")) else np.nan,
                sx1=float(s.get("sx1", np.nan)) if pd.notna(s.get("sx1")) else np.nan,
                qs2=float(s.get("qs2", np.nan)) if pd.notna(s.get("qs2")) else np.nan,
                dnl2=float(s.get("dnl2", np.nan)) if pd.notna(s.get("dnl2")) else np.nan,
                qsx2=float(s.get("qsx2", np.nan)) if pd.notna(s.get("qsx2")) else np.nan,
                sx2=float(s.get("sx2", np.nan)) if pd.notna(s.get("sx2")) else np.nan,
                phqd=float(s.get("phqd", np.nan)) if pd.notna(s.get("phqd")) else np.nan,
                lsqd=float(s.get("lsqd", np.nan)) if pd.notna(s.get("lsqd")) else np.nan,
            )

        r1 = to_row(prev)
        r2 = to_row(prevprev) if prevprev is not None else None
        return r1, r2

    # ========== 下单辅助 ==========
    def _base_size(self) -> int:
        return max(int(self.fixed_size), 1)

    def _max_units(self) -> int:
        return max(int(self.max_multiplier), 1)

    def _flatten_all(self, price: float):
        """以给定价格平掉所有持仓"""
        if self.pos > 0:
            self.sell(price, volume=abs(self.pos))
        elif self.pos < 0:
            self.cover(price, volume=abs(self.pos))
        self.units = 0
        self.mode = "flat"

    def _open_long_units(self, add_units: int, price_open: float, limit_prices: List[float]):
        """做多：第一倍可用开盘，其余用限价"""
        add_units = int(add_units)
        if add_units <= 0:
            return
        size_per_unit = self._base_size()
        # 首次优先用开盘（符合“开盘低于首台阶则以开盘加一次”）
        used_open = False
        if price_open is not None and add_units > 0:
            self.buy(price_open, volume=size_per_unit)
            add_units -= 1
            used_open = True
        # 余下用限价台阶（由低到高依次）
        for lp in limit_prices:
            if add_units <= 0:
                break
            self.buy(lp, volume=size_per_unit)
            add_units -= 1

    def _open_short_units(self, add_units: int, price_open: float, limit_prices: List[float]):
        """做空：第一倍可用开盘，其余用限价"""
        add_units = int(add_units)
        if add_units <= 0:
            return
        size_per_unit = self._base_size()
        used_open = False
        if price_open is not None and add_units > 0:
            self.short(price_open, volume=size_per_unit)
            add_units -= 1
            used_open = True
        for lp in limit_prices:
            if add_units <= 0:
                break
            self.short(lp, volume=size_per_unit)
            add_units -= 1

    # ========== 主体逻辑 ==========
    def on_bar(self, bar: BarData):
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        # 取昨日/前日特征（按当前bar时间向前）
        f_prev, f_prevprev = self._get_feat_prev_and_prevprev(bar.datetime)
        if f_prev is None:
            self.put_event()
            return

        # 方向信号（昨日）
        long_allowed = (int(f_prev.qsx2) == 1)
        short_allowed = (int(f_prev.sx2) == -1)
        long_prev = int(f_prevprev.qsx2) == 1 if f_prevprev else False
        short_prev = int(f_prevprev.sx2) == -1 if f_prevprev else False

        # 可选 lj 过滤
        def pass_lj():
            return (not self.use_lj_filter) or (
                not np.isnan(f_prev.lj) and f_prev.lj >= float(self.lj_threshold)
            )

        # 1) —— 区间起始（以开盘切换+建仓一倍）
        if long_allowed and not long_prev:
            # 新的做多区间开启
            self._flatten_all(bar.open_price)
            if pass_lj():
                self.buy(bar.open_price, self._base_size())
                self.units = 1
                self.mode = "long"
                self.write_log(f"做多区间开启@{bar.open_price}，units=1")

        if short_allowed and not short_prev:
            # 新的做空区间开启
            self._flatten_all(bar.open_price)
            if pass_lj():
                self.short(bar.open_price, self._base_size())
                self.units = 1
                self.mode = "short"
                self.write_log(f"做空区间开启@{bar.open_price}，units=1")

        # 2) —— 区间终止（以开盘平仓）
        if self.mode == "long" and int(f_prev.qsx2) == 0:
            if self.pos > 0:
                self.sell(bar.open_price, abs(self.pos))
            self.units = 0
            self.mode = "flat"
            self.write_log(f"做多区间结束@{bar.open_price}")

        if self.mode == "short" and int(f_prev.sx2) == 0:
            if self.pos < 0:
                self.cover(bar.open_price, abs(self.pos))
            self.units = 0
            self.mode = "flat"
            self.write_log(f"做空区间结束@{bar.open_price}")

        # 3) —— 加仓逻辑（最多到 max_multiplier）
        max_units = self._max_units()
        # 做多区间
        if self.mode == "long" and long_allowed and pass_lj():
            # 昨日台阶：sx1, dnl1
            if not np.isnan(f_prev.sx1) and not np.isnan(f_prev.dnl1) and f_prev.dnl1 > 0:
                thresholds = [f_prev.sx1 - k * f_prev.dnl1 for k in range(1, max_units)]  # L1..L4
                desired = 1  # 初始1倍
                # A) 开盘下破首台阶 → 至少2倍
                if bar.open_price <= thresholds[0]:
                    desired = max(desired, 2)
                # B) 盘中最低触及更深台阶
                crossed = sum(bar.low_price <= th for th in thresholds)
                desired = max(desired, 1 + crossed)
                desired = min(desired, max_units)

                if desired > self.units:
                    to_add = desired - self.units
                    # 第一倍（若还没>=2且满足开盘条件）用开盘，其余用限价台阶（从靠近的开始）
                    # 选择用于限价的台阶价：从 L(self.units) 之后的价格
                    start_idx = self.units  # 已持有n倍，下一目标是第 n+1 个台阶
                    limit_prices = thresholds[start_idx: start_idx + to_add]
                    # 如果开盘条件不满足，就全部用限价；若满足，open会先占掉1倍
                    use_open = (bar.open_price <= thresholds[0]) and (self.units < 2)
                    self._open_long_units(
                        add_units=to_add,
                        price_open=bar.open_price if use_open else None,
                        limit_prices=limit_prices
                    )
                    self.units += to_add

        # 做空区间（对称）
        if self.mode == "short" and short_allowed and pass_lj():
            if not np.isnan(f_prev.qsx1) and not np.isnan(f_prev.dnl1) and f_prev.dnl1 > 0:
                thresholds = [f_prev.qsx1 + k * f_prev.dnl1 for k in range(1, max_units)]  # U1..U4
                desired = 1
                if bar.open_price >= thresholds[0]:
                    desired = max(desired, 2)
                crossed = sum(bar.high_price >= th for th in thresholds)
                desired = max(desired, 1 + crossed)
                desired = min(desired, max_units)

                if desired > self.units:
                    to_add = desired - self.units
                    start_idx = self.units
                    limit_prices = thresholds[start_idx: start_idx + to_add]
                    use_open = (bar.open_price >= thresholds[0]) and (self.units < 2)
                    self._open_short_units(
                        add_units=to_add,
                        price_open=bar.open_price if use_open else None,
                        limit_prices=limit_prices
                    )
                    self.units += to_add

        # 缓存昨日信号便于监控
        self.last_qsx2 = int(f_prev.qsx2) if not np.isnan(f_prev.qsx2) else 0
        self.last_sx2  = int(f_prev.sx2)  if not np.isnan(f_prev.sx2)  else 0
        self.put_event()
