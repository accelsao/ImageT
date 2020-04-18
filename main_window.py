import cv2
from PyQt5.QtGui import QImage

from PyQt5.QtWidgets import QMainWindow, QMenuBar, QFileDialog
from PyQt5.QtCore import QSize

from main_board import MainBoard

class MainWindow(QMainWindow):
    def __init__(self, height, width):
        super(MainWindow, self).__init__()
        self.setFixedSize(QSize(width, height))

        self.main_board = MainBoard()

        main_menubar = QMenuBar()
        main_menubar.addAction('Upload', self.uploadImage)
        main_menubar.addAction('Save', self.saveImage)

        self.setMenuBar(main_menubar)
        self.setCentralWidget(self.main_board)

    def uploadImage(self):
        filename, tmp = QFileDialog.getOpenFileName(
            self, caption='Open Image', directory='./images', filter='*.png *.jpg *.bmp')

        if filename is None:
            return

        src = cv2.imread(filename)
        self.main_board.uploadImage(src)

    def saveImage(self):
        filename, tmp = QFileDialog.getSaveFileName(
            self, caption='Save Image', directory='./images', filter='*.png *.jpg *.bmp')
        self.main_board.saveImage(filename=filename)
