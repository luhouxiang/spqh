from __future__ import annotations
import pandas as pd
from typing import Optional, Iterable

COMMON_DT_NAMES = {"datetime","time","dt","timestamp","trade_time","bar_time"}

def normalize_datetime_columns(df: pd.DataFrame, prefer: Optional[Iterable[str]] = None, tz: Optional[str] = None) -> pd.DataFrame:
    """
    Ensure all datetime-like columns are converted to pandas datetime64[ns], so scalars come out as pd.Timestamp.
    - prefer: a list or set of column names to treat first (default includes common names).
    - tz: if provided, localize naive times to this tz (no conversion), otherwise leave tz-naive.
    """
    if df is None or len(df.columns) == 0:
        return df
    cols = list(prefer) if prefer else list(COMMON_DT_NAMES)
    cols = [c for c in cols if c in df.columns]
    # Also include any columns already dtype datetime-like
    cols += [c for c in df.columns if str(df[c].dtype).startswith("datetime") and c not in cols]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        out[c] = pd.to_datetime(out[c], errors="coerce")
        if tz is not None:
            # if naive -> localize; if tz-aware, convert to the target tz
            if getattr(out[c].dt, "tz", None) is None:
                out[c] = out[c].dt.tz_localize(tz)
            else:
                out[c] = out[c].dt.tz_convert(tz)
    return out
