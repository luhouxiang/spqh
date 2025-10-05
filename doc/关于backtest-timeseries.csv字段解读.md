BacktestingEngine.calculate_result() 产物）：

* close_price：该根K线的收盘价，用于持仓估值、部分成交价计算的基准。

* pre_close：上一根K线的收盘价（用于计算持仓盈亏的基准）。

* trades：当根K线内发生的成交记录列表（保存为字符串）。想要还原为结构化数据可用 ast.literal_eval 解析，再逐条看 price/volume/direction/time。

* trade_count：当根K线内的成交笔数（trades 的长度）。

* start_pos：本根K线开始时的净持仓（>0 多仓手数，<0 为空仓手数，单位：手）。

* end_pos：本根K线结束时的净持仓（单位：手）。

* turnover：当根K线内的成交额（资金换手），≈ Σ(|成交价| × 成交手数 × 合约乘数）。

* commission：当根K线内的手续费，≈ turnover × 手续费率rate。

* slippage：当根K线内的滑点成本，≈ Σ(滑点tick数 × 最小价跳pricetick × 合约乘数 × 成交手数)。

* trading_pnl（已实现盈亏）：本根K线内因为平仓产生的盈亏汇总（实现部分）。

* holding_pnl（持仓盈亏/浮盈浮亏）：因价格从 pre_close 变到 close_price，对持有头寸造成的估值变化。
常见近似：(close_price - pre_close) × start_pos × 合约乘数（多头为正、空头相反）。

* total_pnl：总盈亏 = trading_pnl + holding_pnl。

* net_pnl：净盈亏 = total_pnl - commission - slippage（扣除手续费和滑点后的实际贡献）。

* balance：账户权益/资金曲线（逐根滚动），balance_t = balance_{t-1} + net_pnl_t。

* return：该根的收益率，≈ net_pnl / 上一根balance（小数或%视实现；你统计里 max_ddpercent 是 -10.73，说明 drawdown 用 % 表达，return 多在小数制）。

* highlevel：迄今为止的资金曲线峰值（High Water Mark）。

* drawdown：本根资金曲线相对峰值的回撤额（≤0，单位同资金，通常为“金额”）。

* ddpercent：本根回撤比例（%）。≈ drawdown / highlevel × 100（负数代表下行，例：-10.7 即 -10.7%）。

常见关系/公式小抄

* total_pnl = trading_pnl + holding_pnl

* net_pnl = total_pnl - commission - slippage

* balance_t = balance_{t-1} + net_pnl_t

* return_t ≈ net_pnl_t / balance_{t-1}

* highlevel_t = max(highlevel_{t-1}, balance_t)

* drawdown_t = balance_t - highlevel_t (≤ 0)

* ddpercent_t = drawdown_t / highlevel_t × 100