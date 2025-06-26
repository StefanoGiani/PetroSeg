# Python code to load two images and to control using a button what image to display.

# Author Stefano Giani

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageSwitcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Switcher")

        self.image1 = None
        self.image2 = None
        self.image1_size = (0, 0)
        self.image2_size = (0, 0)

        self.label = tk.Label(root)
        self.label.pack()

        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.load_button1 = tk.Button(button_frame, text="Load Image 1", command=self.load_image1)
        self.load_button1.pack(side=tk.LEFT, padx=5)

        self.load_button2 = tk.Button(button_frame, text="Load Image 2", command=self.load_image2)
        self.load_button2.pack(side=tk.LEFT, padx=5)

        self.show_button1 = tk.Button(button_frame, text="Image 1", command=self.show_image1, state=tk.DISABLED)
        self.show_button1.pack(side=tk.LEFT, padx=5)

        self.show_button2 = tk.Button(button_frame, text="Image 2", command=self.show_image2, state=tk.DISABLED)
        self.show_button2.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(root, text="Image size: N/A", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)

    def load_image1(self):
        path = filedialog.askopenfilename()
        if path:
            img = Image.open(path)
            self.image1_size = img.size
            self.image1 = ImageTk.PhotoImage(img)
            self.label.config(image=self.image1)
            self.update_status(self.image1_size)
            self.check_images_loaded()

    def load_image2(self):
        path = filedialog.askopenfilename()
        if path:
            img = Image.open(path)
            self.image2_size = img.size
            self.image2 = ImageTk.PhotoImage(img)
            self.label.config(image=self.image2)
            self.update_status(self.image2_size)
            self.check_images_loaded()

    def check_images_loaded(self):
        if self.image1 and self.image2:
            self.show_button1.config(state=tk.NORMAL)
            self.show_button2.config(state=tk.NORMAL)

    def show_image1(self):
        if self.image1:
            self.label.config(image=self.image1)
            self.update_status(self.image1_size)

    def show_image2(self):
        if self.image2:
            self.label.config(image=self.image2)
            self.update_status(self.image2_size)

    def update_status(self, size):
        self.status_label.config(text=f"Image size: {size[0]} x {size[1]} pixels")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSwitcherApp(root)
    root.mainloop()

