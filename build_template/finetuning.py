import label_template
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget
from PyQt5 import QtCore
import sys
import os
import json
import logging
import time
import shutil
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class Label(QWidget):
    def __init__(self):
        super().__init__()
        self.MainWindow = QMainWindow()
        self.ui = label_template.Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        self.ui.actionOpen.triggered.connect(self.openFile)
        self.ui.SaveButton.clicked.connect(self.save)
        self.last = -1
        self.path = ''
        self.savePath = './save'
        self.backup = r'F:\template'
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.save_button = False  # Save skip

        self.files_num = -1

        self.ui.SaveButton.setEnabled(False)

    def openFile(self):
        openfile_name = QFileDialog.getExistingDirectory(self, "Select dictionary", './')
        self.path = openfile_name
        # print(self.path)
        start_time = time.time()
        while (True):
            QtCore.QCoreApplication.processEvents()
            if self.path != '':
                break
            elif time.time() - start_time > 5:
                self.ui.LogText.setPlainText("You select none!")
                start_time = time.time()

        self.ui.SaveButton.setEnabled(True)

        self.files_num = len(os.listdir(self.path))

        self.start()

    def start(self):
        files = os.listdir(self.path)
        if os.path.exists('log.txt'):
            with open('log.txt') as log:
                logs = log.readlines()
                if len(logs) > 0:
                    self.last = int(logs[-1])
        start = self.last + 1
        start_time = time.time()
        for i in range(start, len(files)):
            save_name = os.path.join(self.savePath, files[i][:-4] + 'json')
            self.ui.LogText.setPlainText(files[i])
            paper = json.load(open(os.path.join(self.path, files[i])))
            abstract_text = paper['abstract_text']
            template = paper['abstract_template']
            old_label = [str(int(each)+1) for each in paper['label'].split()]
            tmp = []
            this_time = time.time()
            print(str((this_time-start_time)/60)[:4])
            start_time = this_time
            for i in range(len(abstract_text)):
                # xx = ' '.join(template[i])
                # xx = xx.replace('XXX',' ')
                # xx = xx + ' #'
                tmp.append(' '.join(abstract_text[i])+'\n'+old_label[i])
            self.ui.Abstract.setPlainText('\n\n\n'.join(tmp) + '\n')
            while (True):
                QtCore.QCoreApplication.processEvents()
                if self.save_button:
                    try:
                        abstract = self.ui.Abstract.toPlainText()
                        abstract = abstract.split('\n')
                        template = []
                        label = ''
                        for each in abstract:
                            if each == '' or each == '\n' or '#' in each: continue
                            each = each.replace('\n', '')
                            if len(each) <= 3:
                                label += str(int(each) - 1) + '\n'
                            else:
                                each = word_tokenize(each)
                                template.append(each)
                        flag = -1
                        for j in range(len(abstract_text)):
                            for k in range(len(abstract_text[j])):
                                if k >= len(template[j]) or abstract_text[j][k] != template[j][k]:
                                    template[j].insert(k, "XXX")
                            if len(abstract_text[j]) != len(template[j]):
                                flag = j
                                break
                        paper['abstract_template'] = template
                        label=label.replace('5','4')
                        label=label.replace('6','5')
                        paper['label'] = label
                        tmp = label.split('\n')[:-1]
                        if flag >= 0:
                            self.save_button = False
                            self.ui.LogText.setPlainText("Wrong labeling: %d!" % flag)
                            continue
                        if len(template) != len(tmp):
                            self.save_button = False
                            self.ui.LogText.setPlainText("You skip label some sentence!")
                            continue
                        json.dump(paper, open(save_name, 'w'))
                        shutil.copy(save_name, self.backup)
                        self.save_button = False
                        self.ui.Abstract.clear()
                        break
                    except Exception as e:
                        print(e)
                        with open('temp','w',encoding='utf-8') as f:
                            f.write(self.ui.Abstract.toPlainText())
                        self.ui.LogText.setPlainText("Exception catch!")
                        self.save_button = False
                        continue


            self.last += 1
            self.ui.DoneText.setPlainText("Done: %d of %d" % (len(os.listdir(self.savePath)), self.files_num))
            logger.info("%d" % self.last)

    def save(self):
        # skip curren abstract
        self.save_button = True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Label = Label()
    Label.MainWindow.show()
    sys.exit(app.exec_())
