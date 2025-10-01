import os
from common.config import LOG_PATH


def create_log_sub_path(path_name: str):
    """
    检查并创建需要保存的日志子目录
    :param path_name: 子目录名字
    :return: None
    """
    sub_path = os.path.join(LOG_PATH, path_name)
    os.makedirs(sub_path, exist_ok=True)
    return sub_path