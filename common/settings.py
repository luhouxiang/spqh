# settings.py
from pathlib import Path

# 如你已有 common.config.DATA_PATH，这里兼容读取；否则退回本地相对路径
try:
    from common.config import DATA_PATH as _DATA_PATH
    DATA_PATH = Path(_DATA_PATH)
except Exception:
    DATA_PATH = Path(__file__).parent

EXPORTS_SUBDIR = "Exports"
OUTPUT_ROOT = DATA_PATH / "backtest_output"
TIMEZONE = "Asia/Shanghai"

# 当前要回测的标的键
TARGET = "agl9"

# 字典配置（依然支持多个标的）
CONFIG = {
    "agl9": {
        "csv": "spqhagl9.csv",      # 会自动补 volume/open_interest
        "exchange": "SHFE",
        "interval": "1d",           # 留空则自动猜
        "start": "2024-07-11",
        "end":   "2024-09-25",
        "rate": 0.00025,
        "slippage": 1,
        "size": 15,
        "pricetick": 1,
        "capital": 1_000_000,
        "strategy": {"fast_window": 10, "slow_window": 30, "fixed_size": 1},
    }
}

CONFIG2 = {
    "agl9": {
        "csv": "spqhagl9.csv",      # 会自动补 volume/open_interest
        "exchange": "SHFE",
        "interval": "1d",           # 留空则自动猜
        "start": "2023-05-10",
        "end":   "2025-08-29",
        "rate": 0.00025,
        "slippage": 1,
        "size": 15,
        "pricetick": 1,
        "capital": 1_000_000,
        "strategy": {"fast_window": 10, "slow_window": 30, "fixed_size": 1},
    },
    "apl9": {
        "csv": "spqhapl9.csv",  # 会自动补 volume/open_interest
        "exchange": "CZCE",
        "interval": "1d",  # 留空则自动猜
        "start": "2023-05-10",
        "end": "2025-08-29",
        "rate": 0.00025,
        "slippage": 1,
        "size": 15,
        "pricetick": 1,
        "capital": 1_000_000,
        "strategy": {"fast_window": 10, "slow_window": 30, "fixed_size": 1},
    }
}