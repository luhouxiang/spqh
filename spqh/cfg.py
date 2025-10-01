#! -*- coding: utf-8 -*-
import yaml
from common.config import CONF_PATH
from common.config import Cfg
import os

CFG_FN = os.path.join(CONF_PATH, 'a1_kline_chart', 'a6_fen_xing', 'a6_fen_xing.yaml')
g_cfg = Cfg(CFG_FN)
