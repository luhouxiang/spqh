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


CURR_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
WORK_PATH = str(CURR_PATH.parent)
LOG_PATH = os.path.join(WORK_PATH, "logs")
CONF_PATH = os.path.join(WORK_PATH, "conf")


@Singleton
class Cfg():
    def __init__(self, path):
        self.conf_file = path
        self.conf = {}

    def load_yaml(self):
        logging.info(f"[conf_file]: {self.conf_file}")
        logging.info(f"[CONF_PATH]: {CONF_PATH}")
        logging.info(f"[curr_path]: {CURR_PATH}")
        logging.info(f"[work_path]: {WORK_PATH}")
        logging.info(f"[LOG_PATH]: {LOG_PATH}")

        try:
            logging.info(f"正在加载配置文件: {self.conf_file}")
            with open(self.conf_file, 'r', encoding='utf-8') as f:
                self.conf = yaml.safe_load(f) or {}
            return self.conf
        except (IOError, yaml.YAMLError) as e:
            logging.error(f"加载配置文件失败: {str(e)}")
            self.conf = {}  # 设置默认空配置
            return self.conf


