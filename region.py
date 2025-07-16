
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

class RegionMarkerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Region Marker App")

        self.image = None
        self.mask = None
        self.tk_image = None

        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.menu = tk.Menu(root)
        self.root.config(menu=self.menu)
        file_menu = tk.Menu(self.menu, tearoff=0)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Save Mask", command=self.save_mask)
        self.menu.add_cascade(label="File", menu=file_menu)

        self.canvas.bind("<Button-1>", self.on_click)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not file_path:
            return
        self.image = Image.open(file_path).convert("RGB")
        self.image_np = np.array(self.image)
        self.mask = np.zeros(self.image_np.shape[:2], dtype=np.uint8)
        self.display_image()

    def display_image(self):
        display_img = self.image.copy()
        self.tk_image = ImageTk.PhotoImage(display_img)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def on_click(self, event):
        if self.image is None:
            return
        x, y = event.x, event.y
        if x >= self.image_np.shape[1] or y >= self.image_np.shape[0]:
            return

        image_bgr = cv2.cvtColor(self.image_np, cv2.COLOR_RGB2BGR)
        mask_floodfill = np.zeros((self.image_np.shape[0]+2, self.image_np.shape[1]+2), np.uint8)

        tolerance = (30, 30, 30)  # BGR tolerance
        cv2.floodFill(image_bgr, mask_floodfill, (x, y), (0, 0, 255),
                      loDiff=tolerance, upDiff=tolerance, flags=4)

        filled_region = mask_floodfill[1:-1, 1:-1]
        self.mask = cv2.bitwise_or(self.mask, filled_region)

        self.show_mask()

    def show_mask(self):
        mask_rgb = np.stack([self.mask]*3, axis=-1) * 255
        mask_img = Image.fromarray(mask_rgb.astype(np.uint8))
        self.tk_image = ImageTk.PhotoImage(mask_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def save_mask(self):
        if self.mask is None:
            messagebox.showerror("Error", "No mask to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png")])
        if file_path:
            mask_img = Image.fromarray((self.mask * 255).astype(np.uint8))
            mask_img.save(file_path)
            messagebox.showinfo("Saved", f"Mask saved to {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RegionMarkerApp(root)
    root.mainloop()
