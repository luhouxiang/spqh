"""
list_xlsx_files.py
------------------
功能：遍历指定文件夹，收集其中所有 .xlsx 文件的完整路径，返回列表。
"""

import os
from typing import List


def list_xlsx(folder: str) -> List[str]:
    """
    遍历文件夹，获取所有 .xlsx 文件的完整路径。

    :param folder: 要遍历的文件夹路径
    :return: 包含所有 .xlsx 文件完整路径的列表
    """
    xlsx_files = []
    for f in os.listdir(folder):
        if f.lower().endswith(".xlsx"):
            xlsx_files.append(os.path.join(folder, f))

    return xlsx_files


if __name__ == "__main__":
    # 示例：运行本文件时，打印结果
    test_folder = r"E:\work\py\spqh\data\exports"  # 换成你的路径
    xlsx_files = list_xlsx(test_folder)
    print("找到的文件：")
    for f in xlsx_files:
        print(f)
