# main_backtest.py
from pathlib import Path
import pandas as pd
from vnpy.trader.constant import Interval, Exchange

from settings import CONFIG, TARGET, DATA_PATH, EXPORTS_SUBDIR, OUTPUT_ROOT, TIMEZONE
from strategies.double_ma import DoubleMaStrategy
from utils_backtest import (
    setup_matplotlib_backend, guess_interval_from_csv, import_csv_to_db, run_backtest_and_output
)


def main():
    # 1) 读配置 settings.py中
    cfg = CONFIG[TARGET]
    symbol = TARGET  # symbol: 'agl9'
    exchange = Exchange(cfg.get("exchange", "SHFE").upper())  # exchange: <Exchange.SHFE: 'SHFE'>
    csv_name = cfg["csv"]  # csv_name: 'spqhagl9.csv'

    # 2) 拼 CSV 路径（优先 data/Exports，其次脚本目录）
    here = Path(__file__).parent  # here: WindowsPath('E:/work/py/spqh/spqh')
    csv_path = (DATA_PATH / EXPORTS_SUBDIR / csv_name)  # csv_path: WindowsPath('E:/work/py/spqh/data/Exports/spqhagl9.csv')
    if not csv_path.exists():
        csv_path = (here / csv_name)
    csv_path = csv_path.resolve()
    assert csv_path.exists(), f"CSV 不存在：{csv_path}"  # csv_path: WindowsPath('E:/work/py/spqh/data/Exports/spqhagl9.csv')

    # 3) 推断 interval & 时间范围
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    dt_series = pd.to_datetime(df["datetime"], errors="raise")

    txt = str(cfg.get("interval","")).lower()
    if txt in ("1d","d","day","daily"): interval = Interval.DAILY
    elif txt in ("60m","1h","h","hour"): interval = Interval.HOUR
    elif txt in ("1m","m","minute"):     interval = Interval.MINUTE
    else:                                 interval = guess_interval_from_csv(dt_series)

    if cfg.get("start") and cfg.get("end"):
        from datetime import datetime
        start_dt = datetime.strptime(cfg["start"], "%Y-%m-%d")
        end_dt   = datetime.strptime(cfg["end"], "%Y-%m-%d")
    else:
        start_dt = pd.to_datetime(dt_series.min()).to_pydatetime()
        end_dt   = pd.to_datetime(dt_series.max()).to_pydatetime()

    # 4) 其它参数
    rate      = float(cfg.get("rate", 2.5/10000))
    slippage  = float(cfg.get("slippage", 1))
    size      = int(cfg.get("size", 10))
    pricetick = float(cfg.get("pricetick", 1))
    capital   = float(cfg.get("capital", 1_000_000))
    params    = dict(cfg.get("strategy", {}))

    vt_symbol = f"{symbol}.{exchange.value}"
    out_dir   = (OUTPUT_ROOT / vt_symbol.replace(".", "_")).resolve()

    print("=== 配置汇总 ===")
    print(f"vt_symbol={vt_symbol}  csv={csv_path}\ninterval={interval.name}  "
          f"start~end={start_dt} ~ {end_dt}\nrate/slip={rate}/{slippage}  "
          f"size/tick={size}/{pricetick}  capital={capital}\nstrategy={params}\n输出目录={out_dir}")

    # 5) 设置 GUI 后端（自动检测）
    interactive = setup_matplotlib_backend(prefer_gui=True)

    # 6) 导入数据库 + 执行回测 + 输出
    import_csv_to_db(csv_path, symbol, exchange, interval, tz=TIMEZONE)
    run_backtest_and_output(
        strategy_cls=DoubleMaStrategy,
        vt_symbol=vt_symbol, interval=interval, start_dt=start_dt, end_dt=end_dt,
        rate=rate, slippage=slippage, size=size, pricetick=pricetick, capital=capital,
        strategy_params=params, out_dir=out_dir, interactive=interactive
    )


if __name__ == "__main__":
    main()
