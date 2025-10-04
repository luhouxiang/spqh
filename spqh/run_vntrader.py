# run_vntrader.py
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp

# 按需引入网关/应用
from vnpy_ctp import CtpGateway
from vnpy_ctastrategy import CtaStrategyApp
from vnpy_ctabacktester import CtaBacktesterApp
# 也可加 DataManager、Portfolio 等

def main():
    qapp = create_qapp()                 # 创建 Qt 应用
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)

    # 注册网关/应用
    main_engine.add_gateway(CtpGateway)
    main_engine.add_app(CtaStrategyApp)
    main_engine.add_app(CtaBacktesterApp)

    # 启动主窗体
    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()
    qapp.exec_()

if __name__ == "__main__":
    main()
