import argparse
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import get_sketch
import start_removeBack as rm
from networks import ResnetGeneratorUGATIT
import torchvision.transforms as transforms
import torch
import os
from utils import tensor2im


class ImageTrans:
    def __init__(self, image_size, device, pretrained_model, n_res):
        super(ImageTrans, self).__init__()
        self.device = device
        self.transform_func = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(image_size, image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        self.generator = ResnetGeneratorUGATIT(3, 3, n_blocks=n_res, light=True)
        self.load_model(pretrained_model)

        self.origin_image = None
        self.fake_image = None

    def load_model(self, filename):
        params = torch.load(os.path.join('pretrained', filename), map_location=self.device)

        net = self.generator
        if isinstance(net, torch.nn.DataParallel):
            net = net.module

        net.load_state_dict(params['genA2B'])

    def computeTranslateImage(self, src):
        real_image = self.transform_func(src).unsqueeze(0)
        real_image = real_image.to(self.device)
        with torch.no_grad():
            fake_image, _, _ = self.generator(real_image)
        self.fake_image = tensor2im(fake_image)


def resizeImg(w, h):
    re_width = 500
    re_height = 350
    ratio = w / h
    if w > h:
        re_height = int(re_width / ratio)
    else:
        re_width = int(re_height * ratio)
    '''
    print(ratio)
    print(w,h)
    print(re_width,re_height)
    '''
    if re_width > 500:
        re_width = 500
        re_height = int(re_width / ratio)
    elif re_height > 350:
        re_height = 350
        re_width = int(re_height * ratio)
    print(re_width, re_height)
    return re_width, re_height


class Application(tk.Frame):
    def __init__(self, args, master=None):
        super(Application, self).__init__(master)

        image_size = args.image_size
        device = args.device
        n_res = args.n_res
        pretrained_model = args.pretrained_model

        self.model = ImageTrans(image_size=image_size, device=device, pretrained_model=pretrained_model, n_res=n_res)

        self.entry_filename = None
        self.style = tk.IntVar()  # 1:去背 2:填色
        self.image_size = image_size
        self.g = 25

        self.init()

    def init(self):

        btn_transfer = tk.Button(self.master, text="轉換", font="微軟正黑體 12", command=self.transfer_click)
        btn_transfer.configure(bg='#2894FF')
        btn_transfer.place(x=8.5 * self.g, y=3 * self.g)

        btn_chooseFile = tk.Button(self.master, text="選擇檔案", font="微軟正黑體 12", command=self.choose_click)
        btn_chooseFile.place(x=8.5 * self.g, y=self.g)

        text_loading = tk.Label(text='', font="微軟正黑體 12")
        text_loading.configure(bg='white')
        text_loading.place(x=11 * self.g, y=3.2 * self.g)

        entry_filename = tk.Entry(self.master, font="微軟正黑體 12", width=18)
        entry_filename.configure(bg='#F5F5F5')
        entry_filename.place(x=self.g, y=self.g + 5)

        lab_before = tk.Label(text='before transfer', font="微軟正黑體 14")
        lab_before.configure(bg='white')
        lab_before.place(x=self.g, y=5 * self.g)

        chk_rmBack = tk.Radiobutton(self.master, text='去背', variable=self.style, font="微軟正黑體 14", value=1)
        chk_rmBack.configure(bg='white')
        chk_rmBack.place(x=self.g, y=3 * self.g)

        chk_paint = tk.Radiobutton(self.master, text='填色', variable=self.style, font="微軟正黑體 14", value=2)
        chk_paint.configure(bg='white')
        chk_paint.place(x=4 * self.g, y=3 * self.g)

        self.text_loading = text_loading
        self.entry_filename = entry_filename

    def choose_click(self):
        file = filedialog.askopenfilename(parent=self.master, initialdir="images",
                                          title='Please select a directory')

        self.entry_filename.insert(0, file)

        open_img = Image.open(file).resize((256, 256))
        # w, h = open_img.size
        # re_width, re_height = resizeImg(w, h)
        # load_img = ImageTk.PhotoImage(open_img.resize((re_width, re_height), Image.ANTIALIAS))

        load_img = ImageTk.PhotoImage(open_img)
        img_before = tk.Label(self.master, image=load_img)
        img_before.image = load_img
        img_before.place(x=self.g, y=6 * self.g)

        # precompute
        self.computeStyle2 = threading.Thread(target=self.model.computeTranslateImage,
                                              args=(open_img,))
        self.computeStyle2.start()
        self.file = file

    def transfer_click(self):
        self.text_loading.configure(text='轉換中...')
        self.master.update()

        if self.style.get() == 1:  # 去背
            rm.rmBack(self.file)  # 裁切後的果存成cut.jpg
            print('style1', self.file)
            get_sketch.sketch('images/cut.jpg')  # 結果存成result.jpg

            lab_before = tk.Label(text='after transfer', font="微軟正黑體 14")
            lab_before.configure(bg='white')
            lab_before.place(x=2 * self.g + self.image_size, y=5 * self.g)
            open_img = Image.open("images/result.jpg").resize((self.image_size, self.image_size))

            load_img = ImageTk.PhotoImage(open_img)
            img_before = tk.Label(self.master, image=load_img)
            img_before.image = load_img
            img_before.place(x=2 * self.g + self.image_size, y=6 * self.g)

        elif self.style.get() == 2:
            self.computeStyle2.join()

            load_img = ImageTk.PhotoImage(image=Image.fromarray(self.model.fake_image))
            img = tk.Label(self.master, image=load_img)
            img.image = load_img
            img.place(x=2 * self.g + self.image_size, y=6 * self.g)

        self.text_loading.configure(text='已完成')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256, help='the size of image')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--pretrained_model', type=str, default='draw2paintV3'
                                                                '-256x_ugatit_idt2500_colorpreserve_epoch_190.pth',
                        help='pretrianed model')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resnet block')

    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("1075x550")
    root.configure(bg='white')
    root.resizable(0, 0)

    app = Application(args=args, master=root)
    app.mainloop()
