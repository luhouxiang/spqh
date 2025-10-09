import pandas as pd
from common.datetime_normalize import normalize_datetime_columns
import os

import pandas as pd
from pathlib import Path

def excel_to_csv(input_file: str, output_file: str):
    """
    将Excel文件转换为CSV文件，指定字段改名，并将日期列转换为标准datetime格式。
    如果没有 volume 列，则新增该列并将其值置为 0。
    """
    # 读取Excel
    df = pd.read_excel(input_file)


    df = normalize_datetime_columns(df, prefer=["datetime"])
    # 映射：中文列名 -> 英文列名
    rename_map = {
        "日期": "datetime",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交价": "close",
        "成交量": "volume",   # 顺手支持中文“成交量”
    }

    # 执行重命名（仅改存在的列）
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 如果存在 datetime 列，转为标准格式
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # 如果没有 volume 列，新增并置为 0
    if "volume" not in df.columns:
        df["volume"] = 0

    # 保存到CSV
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"转换完成: {output_file}")


if __name__ == "__main__":
    # 示例：当前目录下的输入输出
    input_path = r"E:\work\py\spqh\data\Exports\spqhagl9.xlsx"  # 修改为你的Excel路径

    p = Path(input_path)

    filename = p.name  # 'spqhagl9.xlsx'
    stem = p.stem  # 'spqhagl9'  —— 基本名
    suffix = p.suffix  # '.xlsx'     —— 后缀

    upper_dir = p.parent.parent  # 上一层路径：E:\work\py\spqh\data

    output_path = os.path.join(upper_dir, f"{stem}.csv")
    excel_to_csv(input_path, output_path)