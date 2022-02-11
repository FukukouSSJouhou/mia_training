#!/usr/bin/env python3
import sys
from collections import defaultdict

import numpy as np
from PySide2 import QtCore
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import QObject, Slot, Signal
from PySide2.QtQml import QQmlApplicationEngine
from PySide2.QtWidgets import QApplication
from tensorflow.keras import callbacks


class LearningVisualizationCallback(callbacks.Callback):
    """学習曲線を可視化するためのコールバッククラス"""

    def __init__(self, higher_better_metrics, fig=None, ax=None):
        self._metric_histories = defaultdict(list)
        self._metric_history_lines = {}
        self._higher_better_metrics = set(higher_better_metrics)
        self._metric_type_higher_better = defaultdict(bool)
        self._best_score_vlines = {}
        self._best_score_texts = {}


    def on_epoch_end(self, epoch, logs=None):
        """各エポック毎に呼ばれるコールバック"""

        # 各メトリックのスコアを保存する
        for metric, score in logs.items():
            self._metric_histories[metric].append(score)

            # 初回だけの設定
            if epoch == 0:
                # メトリックの種別を保存する
                for higher_better_metric in self._higher_better_metrics:
                    if higher_better_metric in metric:
                        self._metric_type_higher_better[metric] = True
                        break
                # スコアの履歴を描画するオブジェクトを生成する
                #history_line, = self._ax.plot([], [])
                #self._metric_history_lines[metric] = history_line
                #history_line.set_label(metric)
                #if 'val' not in metric:
                    # 学習データのメトリックは検証データに比べると重要度が落ちるので点線
                #    history_line.set_linestyle('--')
                #else:
                    # ベストスコアの線を描画するオブジェクトを生成する
                #    best_vline = self._ax.axvline(0)
                #    best_vline.set_color(history_line.get_color())
                 #   best_vline.set_linestyle(':')
                 #   self._best_score_vlines[metric] = best_vline
                 #   # ベストスコアの文字列を描画するオブジェクトを生成する
                  #  vpos = 'top' if self._metric_type_higher_better[metric] else 'bottom'
                  #  best_text = self._ax.text(0, 0, '',
                #                              va=vpos, ha='right', weight='bold')
                 #   best_text.set_color(history_line.get_color())
                 #   self._best_score_texts[metric] = best_text

        # 描画内容を更新する
        for metric, scores in self._metric_histories.items():
            # グラフデータを更新する
            history_line = self._metric_history_lines[metric]
            history_line.set_data(np.arange(len(scores)), scores)
            if 'val' in metric:
                if self._metric_type_higher_better[metric]:
                    best_score_find_func = np.max
                    best_epoch_find_func = np.argmax
                else:
                    best_score_find_func = np.min
                    best_epoch_find_func = np.argmin
                best_score = best_score_find_func(scores)
                # 縦線
                best_epoch = best_epoch_find_func(scores)
                best_vline = self._best_score_vlines[metric]
                best_vline.set_xdata(best_epoch)
                # テキスト
                best_text = self._best_score_texts[metric]
                best_text.set_text('epoch:{}, score:{:.6f}'.format(best_epoch, best_score))
                best_text.set_x(best_epoch)
                best_text.set_y(best_score)


    def show_until_close(self):
        None

class GUI_Connection(QObject):
    def __init__(self,parent=None):
        super(GUI_Connection,self).__init__(parent)
        self.addkunindex=0
        self.serieskun=None
        self.axisx=None
        self.axisy=None
    @Slot(QtCharts.QXYSeries,QtCharts.QValueAxis,QtCharts.QValueAxis)
    def setserieskun(self,seriesniki,axisx,axisy):
        self.serieskun=seriesniki
        self.axisy=axisy
        self.axisx=axisx
    value_upd=Signal(float)
    @Slot(str)
    def printkun(self,strniki):
        print(strniki)
    #insert_datakun=Signal(float,float)
    #@Slot()
    #def clicked_addkun(self):
        #self.insert_datakun.emit(1,5)
        #self.insert_datakun.emit(2,5)
        #self.insert_datakun.emit(3,9)
    def addlossdatakun(self,indexkun,valuekun):
        self.axisx.setMax(indexkun)
        self.serieskun.append(indexkun,valuekun)
    @Slot()
    def clickedkun(self):
        self.addlossdatakun(self.addkunindex,0.8)
        self.addkunindex=self.addkunindex+1
        self.addlossdatakun(self.addkunindex,0.5)
        self.addkunindex=self.addkunindex+1
        self.addlossdatakun(self.addkunindex,0.3)
        self.addkunindex=self.addkunindex+1
        print(self.addkunindex)
        print(self.axisx.max())
def main():
    app=QApplication(sys.argv)
    connectionkun=GUI_Connection()
    engine=QQmlApplicationEngine()
    context = engine.rootContext()
    context.setContextProperty("mainwinconnect",connectionkun)
    engine.load("qmls/mia_training.qml")
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec_())
if __name__ == '__main__':
    main()
