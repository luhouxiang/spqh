# algo_features_store.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional
import pandas as pd
from common.datetime_normalize import normalize_datetime_columns

# ---------------------------
# 内部：定位 SQLite 路径
# ---------------------------
def _guess_sqlite_path_via_vnpy() -> Optional[Path]:
    """
    尝试通过 vn.py 的 get_database() 推断 SQLite 文件路径。
    若不是 SQLite 或无法推断，返回 None。
    """
    try:
        from vnpy.trader.database import get_database
        db = get_database()
    except Exception:
        return None
    return Path(db.db.database)


def _open_conn(override_path: Optional[Path] = None) -> sqlite3.Connection:
    db_path = _guess_sqlite_path_via_vnpy()
    # sqlite3 会在同库内新建表，不会影响 vn.py 原有表
    return sqlite3.connect(db_path)


# ---------------------------
# 表结构与通用操作
# ---------------------------
def table_name_for_algo(algo_name: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in algo_name.strip().lower())
    return f"algo_{safe}_features"

def ensure_algo_table(
    algo_name: str,
    feature_cols: Iterable[str],
    versioned: bool = True,
    override_path: Optional[Path] = None,
) -> None:
    """
    为算法创建表（若不存在）。主键：symbol, exchange, interval, datetime, version
    """
    tname = table_name_for_algo(algo_name)
    features_sql = ",\n  ".join(f'"{c}" REAL' for c in feature_cols)
    ver_col = ',"version" TEXT DEFAULT \'1\'' if versioned else ""
    pk_cols = '"symbol","exchange","interval","datetime"' + (',"version"' if versioned else "")

    sql = f"""
    CREATE TABLE IF NOT EXISTS "{tname}" (
      "symbol"   TEXT NOT NULL,
      "exchange" TEXT NOT NULL,
      "interval" TEXT NOT NULL,
      "datetime" TEXT NOT NULL
      {ver_col},
      {features_sql},
      PRIMARY KEY ({pk_cols})
    );
    """
    idx_sql = f'CREATE INDEX IF NOT EXISTS "idx_{tname}_seidt" ON "{tname}"("symbol","exchange","interval","datetime");'

    with _open_conn(override_path) as conn:
        conn.execute(sql)
        conn.execute(idx_sql)
        conn.commit()

def upsert_algo_features(
    algo_name: str,
    df: pd.DataFrame,
    symbol: str,
    exchange: str,
    interval: str,
    feature_cols: Iterable[str],
    version: str = "1",
    datetime_col: str = "datetime",
    override_path: Optional[Path] = None,
) -> int:
    """
    批量 UPSERT 中间变量。
    DataFrame 至少包含 datetime_col 与 feature_cols。
    """
    tname = table_name_for_algo(algo_name)

    # 统一列
    need = [datetime_col] + list(feature_cols)
    # 强制将 DataFrame 的 datetime 列标准化为 pandas.Timestamp
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="raise")
    for c in need:
        if c not in df.columns:
            df[c] = pd.NA

    rows = []
    for _, row in df[need].iterrows():
        dt = pd.to_datetime(row[datetime_col]).to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
        values = [symbol, exchange, interval, dt, version] + [
            None if pd.isna(row[c]) else float(row[c]) for c in feature_cols
        ]
        rows.append(values)

    cols = '("symbol","exchange","interval","datetime","version",' + ",".join(f'"{c}"' for c in feature_cols) + ")"
    placeholders = "(" + ",".join(["?"] * (5 + len(feature_cols))) + ")"
    updates = ",".join(f'"{c}"=excluded."{c}"' for c in feature_cols)

    sql = f"""
    INSERT INTO "{tname}" {cols}
    VALUES {placeholders}
    ON CONFLICT("symbol","exchange","interval","datetime","version")
    DO UPDATE SET {updates};
    """

    with _open_conn(override_path) as conn:
        conn.executemany(sql, rows)
        conn.commit()
        return len(rows)

def upsert_algo_features_many(
    algo_name: str,
    items: List[Dict],  # 每个元素：{"df": DataFrame, "symbol": "...", "exchange": "...", "interval": "...", "version": "1", "datetime_col":"datetime"}
    feature_cols: List[str],
    override_path: Optional[Path] = None,
) -> int:
    ensure_algo_table(algo_name, feature_cols, override_path=override_path)
    total = 0
    for it in items:
        total += upsert_algo_features(
            algo_name=algo_name,
            df=it["df"],
            symbol=it["symbol"],
            exchange=it["exchange"],
            interval=it["interval"],
            feature_cols=feature_cols,
            version=it.get("version", "1"),
            datetime_col=it.get("datetime_col", "datetime"),
            override_path=override_path,
        )
    return total

def load_algo_features(
    algo_name: str,
    symbol: str,
    exchange: str,
    interval: str,
    start_dt,
    end_dt,
    feature_cols: Iterable[str],
    version: str = "1",
    override_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    按时间段读取算法特征，返回 index=datetime 的 DataFrame。
    """
    # 注意：本函数将返回 index 为 pandas.DatetimeIndex（Timestamp 标量）
    tname = table_name_for_algo(algo_name)
    start_s = pd.to_datetime(start_dt).strftime("%Y-%m-%d %H:%M:%S")
    end_s   = pd.to_datetime(end_dt).strftime("%Y-%m-%d %H:%M:%S")
    cols = ",".join(f'"{c}"' for c in feature_cols)
    sql = f"""
    SELECT "datetime", {cols}
    FROM "{tname}"
    WHERE "symbol"=? AND "exchange"=? AND "interval"=? AND "version"=?
      AND "datetime" BETWEEN ? AND ?
    ORDER BY "datetime" ASC;
    """

    with _open_conn(override_path) as conn:
        cur = conn.execute(sql, (symbol, exchange, interval, version, start_s, end_s))
        recs = cur.fetchall()

    if not recs:
        return pd.DataFrame(columns=feature_cols).set_index(pd.DatetimeIndex([], name="datetime"))

    out = pd.DataFrame(recs, columns=["datetime"] + list(feature_cols))
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.set_index("datetime")

def load_algo_features_by_symbols(
    algo_name: str,
    symbols: List[Tuple[str, str]],  # [(symbol, exchange), ...]
    interval: str,
    start_dt,
    end_dt,
    feature_cols: List[str],
    version: str = "1",
    override_path: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for symbol, exchange in symbols:
        df = load_algo_features(
            algo_name=algo_name,
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start_dt=start_dt,
            end_dt=end_dt,
            feature_cols=feature_cols,
            version=version,
            override_path=override_path,
        )
        out[f"{symbol}.{exchange}"] = df
    return out

# 便捷：从 CSV 导入
def upsert_from_csv(
    algo_name: str,
    csv_path: Path,
    symbol: str, exchange: str, interval: str,
    feature_cols: Iterable[str],
    version: str = "1",
    datetime_col: str = "datetime",
    tz: Optional[str] = None,
    override_path: Optional[Path] = None,
) -> int:
    df = pd.read_csv(csv_path)

    df = normalize_datetime_columns(df, prefer=["datetime"])
    # 统一小写列名
    df.columns = [c.lower().strip() for c in df.columns]
    if datetime_col.lower() not in df.columns:
        raise ValueError(f"CSV 缺少 {datetime_col}")

    dt = pd.to_datetime(df[datetime_col.lower()], errors="coerce")
    if tz:
        try:
            dt = dt.dt.tz_localize(tz).dt.tz_convert(None)
        except Exception:
            dt = dt.dt.tz_convert(None) if hasattr(dt.dt, "tz_convert") else dt
    df[datetime_col] = dt

    ensure_algo_table(algo_name, feature_cols, override_path=override_path)
    return upsert_algo_features(
        algo_name=algo_name,
        df=df,
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        feature_cols=feature_cols,
        version=version,
        datetime_col=datetime_col,
        override_path=override_path,
    )