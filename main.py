import argparse
import sys

from PyQt5.QtWidgets import QApplication

from main_window import MainWindow

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_window_height', type=int, default=720, help='the height of main window')
    parser.add_argument('--main_window_width', type=int, default=640, help='the width of main window')

    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(height=args.main_window_height, width=args.main_window_width)
    window.show()
    app.exec_()