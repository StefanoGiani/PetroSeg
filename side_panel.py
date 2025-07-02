import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageRGBViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image RGB Viewer")

        # Left frame for RGB info
        self.info_frame = tk.Frame(root, width=200, bg="lightgray")
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.rgb_label = tk.Label(self.info_frame, text="RGB: ", font=("Arial", 14), bg="lightgray")
        self.rgb_label.pack(pady=20)

        self.load_button = tk.Button(self.info_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        # Right frame for image display
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.image_frame, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image = None
        self.tk_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            self.image = Image.open(file_path).convert("RGB")
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.canvas.bind("<Motion>", self.show_rgb)

    def show_rgb(self, event):
        if self.image:
            x, y = event.x, event.y
            if 0 <= x < self.image.width and 0 <= y < self.image.height:
                r, g, b = self.image.getpixel((x, y))
                self.rgb_label.config(text=f"RGB: ({r}, {g}, {b})")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRGBViewer(root)
    root.mainloop()

