# -*- coding: utf-8 -*-

def Singleton(cls):
    """
    实现单例的装饰器
    :param cls:
    :return:
    """
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner