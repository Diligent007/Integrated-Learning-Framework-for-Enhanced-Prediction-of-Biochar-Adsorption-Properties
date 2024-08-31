from PyQt5 import uic
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QMessageBox, QFileDialog, QHeaderView, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import csv
from scipy.fftpack import fft
from scipy.interpolate import interp1d
import qdarkstyle
import matplotlib as mpl
import seaborn as sns
import joblib
import pandas as pd
from sklearn.decomposition import PCA
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

mpl.rcParams['font.sans-serif'] = ['KaiTi']  # 保证正常显示中文
mpl.rcParams['font.serif'] = ['KaiTi']  # 保证正常显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 保证负号正常显示

warnings.filterwarnings("ignore")  # 忽略警告


class Stats:
    def __init__(self):
        # 从文件中加载UI定义
        self.ui = uic.loadUi("GUI共热解碳.ui")

        # 单一预测
        self.fig1 = plt.Figure(facecolor=(25 / 255, 35 / 255, 45 / 255))
        plt.rcParams['text.color'] = 'white'
        # plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.title.set_text('Input Display')
        self.ax1.set_ylabel('Predictive value', fontsize=12, color="w")
        self.ax1.set_xlabel('True value', fontsize=12, color="w")
        self.ax1.tick_params(axis='x', colors='white')
        self.ax1.tick_params(axis='y', colors='white')
        self.ax1.set_facecolor('black')
        self.ax1.grid(linestyle='--', color="w")
        self.canvas1 = FC(self.fig1)
        self.fig1.tight_layout()
        self.ui.verticalLayout_10.addWidget(self.canvas1)

        # 批量预测
        self.fig2 = plt.Figure(facecolor=(25 / 255, 35 / 255, 45 / 255))
        plt.rcParams['text.color'] = 'white'
        # plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.title.set_text('Prediction display')
        self.ax2.set_ylabel('Predictive value', fontsize=12, color="w")
        self.ax2.set_xlabel('Data Number', fontsize=12, color="w")
        self.ax2.tick_params(axis='x', colors='white')
        self.ax2.tick_params(axis='y', colors='white')
        self.ax2.set_facecolor('black')
        self.ax2.grid(linestyle='--', color="w")
        self.canvas2 = FC(self.fig2)
        self.fig2.tight_layout()
        self.ui.verticalLayout_11.addWidget(self.canvas2)

        self.model = QStandardItemModel(0, 2)
        self.model.setHorizontalHeaderLabels(['Data Number', 'Predictive value'])
        self.ui.tableView.setModel(self.model)
        self.ui.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 训练准确率
        self.fig3 = plt.Figure(facecolor=(25 / 255, 35 / 255, 45 / 255))
        plt.rcParams['text.color'] = 'white'
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.set_ylabel('Error rate', fontsize=12, color="w")
        self.ax3.set_xlabel('数据编号', fontsize=12, color="w")
        self.ax3.tick_params(axis='x', colors='white')
        self.ax3.tick_params(axis='y', colors='white')
        self.ax3.set_facecolor('black')
        self.ax3.grid(linestyle='--', color="w")
        self.canvas3 = FC(self.fig3)
        self.fig3.tight_layout()
        self.ui.verticalLayout_12.addWidget(self.canvas3)

        self.ui.pushButton_2.clicked.connect(self.slot_btn_predict)
        self.ui.pushButton.clicked.connect(self.slot_btn_chooseFile1)
        self.ui.pushButton_3.clicked.connect(self.slot_btn_chooseFile2)
        self.ui.pushButton_4.clicked.connect(self.slot_btn_chooseFile3)
        self.ui.pushButton_5.clicked.connect(self.slot_btn_chooseFile4)
        self.ui.pushButton_6.clicked.connect(self.slot_btn_predict_batch)
        self.ui.pushButton_7.clicked.connect(self.slot_btn_chooseFile5)
        self.ui.pushButton_8.clicked.connect(self.slot_btn_chooseFile6)
        self.ui.pushButton_9.clicked.connect(self.slot_btn_chooseFile7)
        self.ui.pushButton_10.clicked.connect(self.slot_btn_train)

    def slot_btn_predict(self):
        try:
            c = float(self.ui.textEdit.toPlainText())
            h = float(self.ui.textEdit_2.toPlainText())
            o = float(self.ui.textEdit_3.toPlainText())
            n = float(self.ui.textEdit_4.toPlainText())
            s = float(self.ui.textEdit_5.toPlainText())
            ash = float(self.ui.textEdit_6.toPlainText())
            vm = float(self.ui.textEdit_7.toPlainText())
            fc = float(self.ui.textEdit_8.toPlainText())
            time = float(self.ui.textEdit_9.toPlainText())
            t = float(self.ui.textEdit_10.toPlainText())
            v = float(self.ui.textEdit_11.toPlainText())
            ph = float(self.ui.textEdit_12.toPlainText())

            path = self.ui.textEdit_14.toPlainText()

            pattern = list()
            pattern.append(c)
            pattern.append(h)
            pattern.append(o)
            pattern.append(n)
            pattern.append(s)
            pattern.append(ash)
            pattern.append(vm)
            pattern.append(fc)
            pattern.append(time)
            pattern.append(t)
            pattern.append(v)
            pattern.append(ph)

            # # 创建 MinMaxScaler 对象
            # scaler = MinMaxScaler()
            #
            # # 在训练集上拟合归一化器并进行归一化
            # scaler.fit(pattern)
            # normalized_data = scaler.transform(pattern)

            model = joblib.load(path)

            predicted_y = model.predict([pattern])

            self.ui.textEdit_13.clear()
            self.ui.textEdit_13.setText(str(predicted_y[0]))

            x1 = np.linspace(0, len(pattern) - 1, num=int(len(pattern)))
            self.ax1.cla()
            self.ax1.plot(x1, pattern, color=(0, 1, 0))
            self.ax1.title.set_text('输入展示')
            self.ax1.set_ylabel('幅值', fontsize=12, color="w")
            self.ax1.set_xlabel('元素', fontsize=12, color="w")
            self.ax1.tick_params(axis='x', colors='white')
            self.ax1.tick_params(axis='y', colors='white')
            self.ax1.set_facecolor('black')
            self.ax1.grid(linestyle='--', color="w")
            self.canvas1.draw()

        except:
            QMessageBox.about(stats.ui, '程序异常', '''参数输入有误！''')

    def slot_btn_chooseFile1(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.ui, "选取文件")

        if fileName_choose == "":
            return

        self.ui.textEdit_14.clear()
        self.ui.textEdit_14.append(fileName_choose)

    def slot_btn_chooseFile2(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.ui, "选取文件")

        if fileName_choose == "":
            return

        self.ui.textEdit_15.clear()
        self.ui.textEdit_15.append(fileName_choose)

    def slot_btn_chooseFile3(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.ui, "选取文件")

        if fileName_choose == "":
            return

        self.ui.textEdit_16.clear()
        self.ui.textEdit_16.append(fileName_choose)

    def slot_btn_chooseFile4(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.ui, "选取文件")

        if fileName_choose == "":
            return

        self.ui.textEdit_17.clear()
        self.ui.textEdit_17.append(fileName_choose)

    def slot_btn_predict_batch(self):
        try:
            path1 = self.ui.textEdit_15.toPlainText()
            path2 = self.ui.textEdit_16.toPlainText()
            path3 = self.ui.textEdit_17.toPlainText()

            x_data = pd.read_excel(path1, header=None)

            model = joblib.load(path3)

            predicted_y = model.predict(x_data)

            if path2:
                y_data = pd.read_excel(path2, header=None)
                label_predict_sorted = sorted(predicted_y.reshape(len(predicted_y), 1))
                label_train_sorted = sorted(np.array(y_data))
                label_sorted = np.hstack((label_train_sorted, label_predict_sorted))
                np.random.shuffle(label_sorted)
                label_train_sorted = label_sorted[:, 0]
                label_predict_sorted = label_sorted[:, 1]
            else:
                label_predict_sorted = sorted(predicted_y.reshape(len(predicted_y), 1))
                np.random.shuffle(label_predict_sorted)


            x2 = np.linspace(0, len(label_predict_sorted) - 1, num=int(len(label_predict_sorted)))
            self.ax2.cla()
            self.ax2.plot(x2, label_predict_sorted, color=(0, 1, 0))
            if path2:
                self.ax2.plot(x2, np.array(label_train_sorted), color=(1, 0, 0))

            self.ax2.title.set_text('Prediction display')
            self.ax2.set_ylabel('Predictive value', fontsize=12, color="w")
            self.ax2.set_xlabel('Data Number', fontsize=12, color="w")
            self.ax2.tick_params(axis='x', colors='white')
            self.ax2.tick_params(axis='y', colors='white')
            self.ax2.set_facecolor('black')
            self.ax2.grid(linestyle='--', color="w")
            self.canvas2.draw()

            self.model.clear()
            self.model.setHorizontalHeaderLabels(['Data Number', 'Predictive value'])
            for i in range(len(predicted_y)):
                self.model.appendRow([QStandardItem('%d' % (i + 1)), QStandardItem('%f' % predicted_y[i])])

        except:
            QMessageBox.about(stats.ui, '程序异常', '''参数输入有误！''')

    def slot_btn_chooseFile5(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.ui, "选取文件")

        if fileName_choose == "":
            return

        self.ui.textEdit_18.clear()
        self.ui.textEdit_18.append(fileName_choose)

    def slot_btn_chooseFile6(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.ui, "选取文件")

        if fileName_choose == "":
            return

        self.ui.textEdit_19.clear()
        self.ui.textEdit_19.append(fileName_choose)

    def slot_btn_chooseFile7(self):
        fileName_choose = QFileDialog.getExistingDirectory(self.ui, "选取文件")

        if fileName_choose == "":
            return

        self.ui.textEdit_20.clear()
        self.ui.textEdit_20.append(fileName_choose)

    def slot_btn_train(self):
        try:
            path1 = self.ui.textEdit_18.toPlainText()
            path2 = self.ui.textEdit_19.toPlainText()
            path3 = self.ui.textEdit_20.toPlainText()

            # 加载数据
            x_data = pd.read_excel(path1, header=None)
            y_data = pd.read_excel(path2, header=None)

            # 数据预处理
            x_data.columns = [str(i) for i in range(x_data.shape[1])]
            y_data.columns = ['target']

            # 划分数据集，只保留训练集
            X_train, _, y_train, _ = train_test_split(x_data, y_data['target'], test_size=0.2, random_state=42)

            # 训练 RandomForestRegressor 模型
            RF_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1,
                                             random_state=42)
            RF_model.fit(X_train, y_train)

            # 训练 CatBoostRegressor 模型
            CB_model = CatBoostRegressor(iterations=100, depth=8, learning_rate=0.1, l2_leaf_reg=1, random_state=42)
            CB_model.fit(X_train, y_train)

            # 集成模型
            ensemble_model = VotingRegressor(estimators=[('RF', RF_model), ('CB', CB_model)])
            ensemble_model.fit(X_train, y_train)

            # 评估模型
            y_pred = ensemble_model.predict(X_train)
            mse = mean_squared_error(y_train, y_pred)
            r2 = r2_score(y_train, y_pred)
            print(1)
            # 打印总体MSE和R2
            self.ui.textBrowser.append("Total MSE:%f" % mse)
            self.ui.textBrowser.append("Total R2 Score:%f" % r2)

            label_train_sorted = sorted(np.array(y_train).reshape(len(y_train), 1))
            print(label_train_sorted)
            label_predict_sorted = sorted(y_pred.reshape(len(y_pred), 1))
            print(label_predict_sorted)
            label_sorted = np.hstack((label_train_sorted, label_predict_sorted))
            np.random.shuffle(label_sorted)
            label_train_sorted = label_sorted[:, 0]
            label_predict_sorted = label_sorted[:, 1]

            # 对每一对真实值和预测值计算MRE，并打印
            mre_total = list()
            i = 1
            print(4)
            for actual, predicted in zip(label_train_sorted, label_predict_sorted):
                if actual != 0:
                    mre = np.abs((actual - predicted) / actual)
                    self.ui.textBrowser.append(
                        "The true value of data% d is:%f，the predicted value is：%f，and the error rate is：%f" % (i, actual, predicted,
                                                            mre))
                    mre_total.append(mre)
                else:
                    self.ui.textBrowser.append(
                        "The true value of data% d is:%f，and the predicted value is：%f，MRE: Undefined (actual value is zero)" % (i, actual, predicted))

                i = i + 1

            # 保存模型
            joblib.dump(ensemble_model, '%s/trained_model_RF.pkl' % path3)

            x3 = np.linspace(0, len(mre_total) - 1, num=int(len(mre_total)))
            self.ax3.cla()
            # self.ax3.plot(x3, np.array(y_train), color=(0, 1, 0))
            # self.ax3.plot(x3, np.array(y_pred), color=(1, 0, 0))
            self.ax3.plot(x3, mre_total, color=(1, 0, 0))
            self.ax3.set_ylabel('Error rate', fontsize=12, color="w")
            self.ax3.set_xlabel('数据编号', fontsize=12, color="w")
            self.ax3.tick_params(axis='x', colors='white')
            self.ax3.tick_params(axis='y', colors='white')
            self.ax3.set_facecolor('black')
            self.ax3.grid(linestyle='--', color="w")
            self.canvas3.draw()

        except:
            QMessageBox.about(stats.ui, '程序异常', '''参数输入有误！''')


app = QApplication([])
app.setStyleSheet(qdarkstyle.load_stylesheet())
stats = Stats()
# stats.ui.setWindowTitle('智能诊断程序')
stats.ui.show()
app.exec_()
