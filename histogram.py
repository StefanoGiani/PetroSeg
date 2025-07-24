import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt

class HSVHistogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HSV Histogram Viewer")

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.h_button = tk.Button(root, text="Show H Histogram", command=self.show_h_histogram)
        self.h_button.pack()

        self.s_button = tk.Button(root, text="Show S Histogram", command=self.show_s_histogram)
        self.s_button.pack()

        self.v_button = tk.Button(root, text="Show V Histogram", command=self.show_v_histogram)
        self.v_button.pack()

        self.image = None
        self.hsv_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                messagebox.showerror("Error", "Failed to load image.")
                return
            self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            self.image_label.configure(image=image_tk)
            self.image_label.image = image_tk

    def show_histogram(self, channel_index, title, color):
        if self.hsv_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        channel = self.hsv_image[:, :, channel_index]
        plt.figure()
        plt.hist(channel.ravel(), bins=256, range=(0, 256), color=color)
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    def show_h_histogram(self):
        self.show_histogram(0, "Hue Histogram", "red")

    def show_s_histogram(self):
        self.show_histogram(1, "Saturation Histogram", "green")

    def show_v_histogram(self):
        self.show_histogram(2, "Value Histogram", "blue")

if __name__ == "__main__":
    root = tk.Tk()
    app = HSVHistogramApp(root)
    root.mainloop()
