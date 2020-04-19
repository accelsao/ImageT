import os

import cv2
import torch
import torchvision.transforms as transforms
from PyQt5.QtCore import Qt, QRunnable, QThreadPool
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel

from networks import ResnetGeneratorUGATIT
from utils import tensor2im


class ProcessRunnable(QRunnable):
    def __init__(self, target, args=None):
        QRunnable.__init__(self)
        self.t = target
        self.args = args

    def run(self):
        self.t(*self.args)

    def start(self):
        QThreadPool.globalInstance().start(self)


class MainBoard(QLabel):
    def __init__(self, image_size, device, pretrained_model, n_res):
        super(MainBoard, self).__init__()

        self.setAlignment(Qt.AlignCenter)
        self.origin_image = None
        self.fake_image = None
        self.device = device
        self.transform_func = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        self.generator = ResnetGeneratorUGATIT(3, 3, n_blocks=n_res, light=True)
        self.load_model(pretrained_model)

    def load_model(self, filename):
        params = torch.load(os.path.join('pretrained', filename), map_location=self.device)

        net = self.generator
        if isinstance(net, torch.nn.DataParallel):
            net = net.module

        net.load_state_dict(params['genA2B'])

    def uploadImage(self, src):
        """

        :param src: image (OpenCV mat)
        :return: set image to label for display
        """
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.origin_image = src
        self.fake_image = None
        self.getOriginImage()
        p = ProcessRunnable(target=self.computeTranslateImage, args=(src,))
        p.start()

    def computeTranslateImage(self, src):
        real_image = self.transform_func(src).unsqueeze(0)
        real_image = real_image.to(self.device)
        with torch.no_grad():
            fake_image, _, _ = self.generator(real_image)
        self.fake_image = tensor2im(fake_image)

    def saveImage(self, filename):
        self.pixmap().save(filename)

    def getOriginImage(self):
        self.displayImage(self.origin_image)

    def getTranslateImage(self):
        if self.fake_image is None:
            print("Image is computing ... wait for a second any try again.")
            return

        self.displayImage(self.fake_image)

    def displayImage(self, src):
        """
        display images
        :param src: input image with shape h,w,c datatype = numpy
        :return:
        """
        h, w, c = src.shape
        bytesPerLine = 3 * w
        qImg = QImage(src.copy(), w, h, bytesPerLine, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qImg))
