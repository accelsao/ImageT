import argparse
import sys

from PyQt5.QtWidgets import QApplication

from main_window import MainWindow

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_window_height', type=int, default=720, help='the height of main window')
    parser.add_argument('--main_window_width', type=int, default=640, help='the width of main window')
    parser.add_argument('--image_size', type=int, default=256, help='the size of image')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--pretrained_model', type=str, default='draw2paintV3'
                                                                '-256x_ugatit_idt2500_colorpreserve_epoch_190.pth',
                        help='pretrianed model')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resnet block')


    args = parser.parse_args()

    app = QApplication(sys.argv)
    # window = MainWindow(height=args.main_window_height, width=args.main_window_width, image_size=args.image_size, device=args.device, pretrained_model=args.pretrained_model)
    window = MainWindow(args)
    window.show()
    app.exec_()