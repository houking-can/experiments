# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'label_template.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1128, 812)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../.designer/backup/resource/logo.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("font: 75 13pt \"Consolas\";")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Abstract = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.Abstract.setGeometry(QtCore.QRect(60, 120, 1021, 551))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.Abstract.setFont(font)
        self.Abstract.setStyleSheet("background-color:rgb(170, 227, 128)\n"
"")
        self.Abstract.setPlainText("")
        self.Abstract.setObjectName("Abstract")
        self.SaveButton = QtWidgets.QPushButton(self.centralwidget)
        self.SaveButton.setGeometry(QtCore.QRect(490, 690, 111, 61))
        self.SaveButton.setStyleSheet("background-color:rgb(199, 134, 255)")
        self.SaveButton.setObjectName("SaveButton")
        self.Abstract_label = QtWidgets.QLabel(self.centralwidget)
        self.Abstract_label.setGeometry(QtCore.QRect(60, 80, 121, 31))
        self.Abstract_label.setObjectName("Abstract_label")
        self.LogText = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.LogText.setGeometry(QtCore.QRect(150, 30, 461, 41))
        self.LogText.setObjectName("LogText")
        self.Title_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.Title_label_2.setGeometry(QtCore.QRect(150, 0, 72, 31))
        self.Title_label_2.setObjectName("Title_label_2")
        self.DoneText = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.DoneText.setGeometry(QtCore.QRect(660, 30, 301, 41))
        self.DoneText.setObjectName("DoneText")
        self.Title_label_3 = QtWidgets.QLabel(self.centralwidget)
        self.Title_label_3.setGeometry(QtCore.QRect(660, -10, 101, 41))
        self.Title_label_3.setObjectName("Title_label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1128, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.menuFile.addAction(self.actionOpen)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Academic paper abstract sentence label tool"))
        self.SaveButton.setText(_translate("MainWindow", "Save"))
        self.Abstract_label.setText(_translate("MainWindow", "Abstract"))
        self.Title_label_2.setText(_translate("MainWindow", "Log"))
        self.Title_label_3.setText(_translate("MainWindow", "Done"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))

