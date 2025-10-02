import os
import re
import win32com.client as win32


def access_export_excel(accdb= r"E:\work\py\spqh\data\gzspqh17.accdb"):
    export_dir = os.path.join(os.path.dirname(accdb), "Exports")
    os.makedirs(export_dir, exist_ok=True)
    access = win32.Dispatch("Access.Application")
    # access.Visible = False  # 隐藏界面
    # access.UserControl = False  # 防止前台干扰
    try:
        access.OpenCurrentDatabase(accdb)

        # 迭代表
        db = access.CurrentDb()
        for tdf in db.TableDefs:
            name = tdf.Name
            if name.startswith(("MSys", "USys", "~TMP")):
                continue
            safe = re.sub(r'[\/\\\:\*\?\"\<\>\|]', '_', name)
            xlsx = os.path.join(export_dir, f"{safe}.xlsx")
            if os.path.exists(xlsx):
                os.remove(xlsx)
            # acExport=1, acSpreadsheetTypeExcel12Xml=10
            access.DoCmd.TransferSpreadsheet(1, 10, name, xlsx, True)
    finally:
        access.CloseCurrentDatabase()
        access.Quit()


access_export_excel()
