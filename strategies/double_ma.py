# strategies/double_ma.py
from vnpy_ctastrategy import CtaTemplate
from vnpy.trader.object import BarData
from vnpy.trader.utility import ArrayManager

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

    def on_init(self):
        self.write_log("策略初始化")
        self.load_bar(max(self.fast_window, self.slow_window) + 50)

    def on_start(self):
        self.write_log("策略启动")

    def on_stop(self):
        self.write_log("策略停止")

    def on_bar(self, bar: BarData):
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
            if cross_up:  self.buy(bar.close_price, self.fixed_size)
            elif cross_dn: self.short(bar.close_price, self.fixed_size)
        elif self.pos > 0 and cross_dn:
            self.sell(bar.close_price, abs(self.pos))
            self.short(bar.close_price, self.fixed_size)
        elif self.pos < 0 and cross_up:
            self.cover(bar.close_price, abs(self.pos))
            self.buy(bar.close_price, self.fixed_size)

        self.put_event()
