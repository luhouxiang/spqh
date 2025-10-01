#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common.logging_cfg import SysLogInit
from cfg import g_cfg

if __name__ == '__main__':
    SysLogInit('abc', "logs")
    g_cfg.load_yaml()   # 默认最先加载配置文件
