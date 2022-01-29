#!/usr/bin/env python3
import sys

from PySide2.QtCore import QObject
from PySide2.QtQml import QQmlApplicationEngine
from PySide2.QtWidgets import QApplication
class GUI_Connection(QObject):
    def __init__(self,parent=None):
        super(GUI_Connection,self).__init__(parent)


def main():
    app=QApplication(sys.argv)
    connectionkun=GUI_Connection()
    engine=QQmlApplicationEngine()
    context = engine.rootContext()
    context.setContextProperty("mainwinconnect",connectionkun)
    engine.load("mia_training.qml")
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec_())
if __name__ == '__main__':
    main()
