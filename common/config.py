#!/usr/bin/env python3
# encoding: utf-8
"""
@author: lyl
@license: (C) Copyright 2017-2023, Dztec right.
@desc:这个文件主要定义一些全局的目录、或者是文件定义信息
"""
import os
from pathlib import Path
import yaml
from common.utils.singleton import Singleton
import logging


real_path = os.path.dirname(os.path.realpath(__file__))
curr_path = Path(real_path)
work_path = str(curr_path.parent)
LOG_PATH = os.path.join(work_path, "logs")
CONF_PATH = os.path.join(work_path, "conf")

print("[real_path]: ", real_path)
print("[curr_path]: ", curr_path)
print("[work_path]: ", work_path)
print("[LOG_PATH]: ", LOG_PATH)
print("[CONF_PATH]: ", CONF_PATH)
# 计算中心端口
CALC_CENTER_HTTP_PORT = 30238

@Singleton
class Cfg():
    def __init__(self, path):
        self.path = path
        self.conf = {}

    def load_yaml(self):
        try:
            logging.info(f"正在加载配置文件: {self.path}")
            with open(self.path, 'r', encoding='utf-8') as f:
                self.conf = yaml.safe_load(f) or {}
            return self.conf
        except (IOError, yaml.YAMLError) as e:
            logging.error(f"加载配置文件失败: {str(e)}")
            self.conf = {}  # 设置默认空配置
            return self.conf


