import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer with Zoom, Coordinates, and Drag")

        # Left panel for coordinates
        self.left_panel = tk.Frame(root, width=150)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.coord_label = tk.Label(self.left_panel, text="X: \nY: ", font=("Arial", 12))
        self.coord_label.pack(pady=10)

        # Right panel for image and controls
        self.right_panel = tk.Frame(root)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas with scrollbars
        self.canvas = tk.Canvas(self.right_panel, bg='gray')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.hbar = tk.Scrollbar(self.right_panel, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar = tk.Scrollbar(self.right_panel, orient=tk.VERTICAL, command=self.canvas.yview)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.configure(yscrollincrement='2')
        self.canvas.configure(xscrollincrement='2')
        # Buttons
        self.button_frame = tk.Frame(self.left_panel)
        self.button_frame.pack(pady=20)

        tk.Button(self.button_frame, text="Load Image", command=self.load_image).pack(pady=5)
        tk.Button(self.button_frame, text="Zoom In", command=lambda: self.zoom(1.2)).pack(pady=5)
        tk.Button(self.button_frame, text="Zoom Out", command=lambda: self.zoom(0.8)).pack(pady=5)

        # Bind mouse motion and drag
        self.canvas.bind("<Motion>", self.update_coordinates)
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.do_drag)

        # Image state
        self.image = None
        self.tk_image = None
        self.zoom_factor = 1.0
        self.image_id = None
        self.drag_start_x = 0
        self.drag_start_y = 0

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            self.image = Image.open(file_path)
            self.zoom_factor = 1.0
            self.display_image()

    def display_image(self):
        if self.image:
            resized = self.image.resize(
                (int(self.image.width * self.zoom_factor), int(self.image.height * self.zoom_factor)),
                Image.LANCZOS
            )
            self.tk_image = ImageTk.PhotoImage(resized)

            self.canvas.delete("all")
            self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            

    def zoom(self, factor):
        if self.image:
            self.zoom_factor *= factor
            self.display_image()

    def update_coordinates(self, event):
        if self.image:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)

            img_x = int(canvas_x / self.zoom_factor)
            img_y = int(canvas_y / self.zoom_factor)

            if 0 <= img_x < self.image.width and 0 <= img_y < self.image.height:
                self.coord_label.config(text=f"X: {img_x}\nY: {img_y}")
            else:
                self.coord_label.config(text="X: \nY: ")

    def start_drag(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def do_drag(self, event):
        dx = self.drag_start_x - event.x
        dy = self.drag_start_y - event.y
        self.canvas.xview_scroll(int(dx), "units")
        self.canvas.yview_scroll(int(dy), "units")
        self.drag_start_x = event.x
        self.drag_start_y = event.y

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.geometry("800x600")
    root.mainloop()
