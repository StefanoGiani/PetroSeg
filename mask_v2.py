# Python app to create a mask from an image and visualise it.
# This version fill in the mask with the pixels of similar colours compared to where you click.


# Author Stefano Giani


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageMaskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image and Mask Viewer")

        self.image = None
        self.mask = None
        self.display_image = None

        self.canvas = tk.Label(root)
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        load_btn = tk.Button(btn_frame, text="Load Image", command=self.load_image)
        load_btn.grid(row=0, column=0)

        mask_btn = tk.Button(btn_frame, text="Generate Mask", command=self.generate_mask)
        mask_btn.grid(row=0, column=1)

        show_image_btn = tk.Button(btn_frame, text="Image", command=self.show_image)
        show_image_btn.grid(row=0, column=2)

        show_mask_btn = tk.Button(btn_frame, text="Mask", command=self.show_mask)

        self.canvas.bind("<Button-1>", self.on_click)

        show_mask_btn.grid(row=0, column=3)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image = self.image
            self.show_image()

    def generate_mask(self):
        if self.image is not None:
            self.mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
            self.show_mask()

    def show_image(self):
        if self.image is not None:
            self.display_image = self.image
            self.update_display()

    def show_mask(self):
        if self.mask is not None:
            mask_rgb = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB)
            self.display_image = mask_rgb
            self.update_display()

    def update_display(self):
        img = Image.fromarray(self.display_image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

    def on_click(self, event):
        if self.image is None:
            return

        # Convert click coordinates to image coordinates
        img = Image.fromarray(self.image)
        width, height = img.size
        x = int(event.x * self.image.shape[1] / width)
        y = int(event.y * self.image.shape[0] / height)

        # Get the color at the clicked point
        clicked_color = self.image[y, x]
        print(clicked_color)

        # Define a color threshold
        lower = np.clip(clicked_color - 10, 0, 255)
        upper = np.clip(clicked_color + 10, 0, 255)
        print(lower)
        print(upper)


        # Create a mask for similar colors
        mask = cv2.inRange(self.image, lower, upper)
        self.mask = mask
        self.show_mask()


root = tk.Tk()
app = ImageMaskApp(root)
root.mainloop()

