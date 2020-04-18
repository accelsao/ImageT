from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt

class MainBoard(QLabel):
    def __init__(self):
        super(MainBoard, self).__init__()

        self.setAlignment(Qt.AlignCenter)
        self.origin_image = None

    def uploadImage(self, src):
        """

        :param qImg: image (QImage)
        :return: set image to label for display
        """
        self.origin_image = src
        h, w, c = src.shape
        bytesPerLine = 3 * w
        qImg = QImage(src.data, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.setPixmap(QPixmap.fromImage(qImg))

    def saveImage(self, filename):
        self.pixmap().save(filename)
