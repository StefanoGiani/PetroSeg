# Python code to create a window to display an image using tkinter.
# In this version of the code it is possible to zoom in and out of the image.
# In this version, the zoom can also be controlled by the keys + and -.

# Author Stefano Giani

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Class implementing the app
class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer with Zoom")

        self.zoom_level = 1.0
        self.original_image = None

        self.btn_open = tk.Button(root, text="Open Image", command=self.open_image)
        self.btn_open.pack(pady=5)

        self.btn_zoom_in = tk.Button(root, text="Zoom In", command=self.zoom_in)
        self.btn_zoom_in.pack(pady=5)

        self.btn_zoom_out = tk.Button(root, text="Zoom Out", command=self.zoom_out)
        self.btn_zoom_out.pack(pady=5)

        self.label = tk.Label(root)
        self.label.pack()
        
        # Bind + and - keys for zooming
        self.root.bind("<plus>", lambda event: self.zoom_in())
        self.root.bind("<minus>", lambda event: self.zoom_out())
        self.root.bind("<KeyPress-equal>", lambda event: self.zoom_in())  # For '+' without Shift
        self.root.bind("<KeyPress-minus>", lambda event: self.zoom_out())  # For '-' with Shift
        
    # Routine to run in response to the button open image been pressed 
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.original_image = Image.open(file_path)
            self.zoom_level = 1.0
            self.display_image()
            
    # Routine to display the image with the correct level of zooming
    def display_image(self):
        if self.original_image:
            width, height = self.original_image.size
            new_size = (int(width * self.zoom_level), int(height * self.zoom_level))
            resized = self.original_image.resize(new_size, Image.LANCZOS) # Resize the image using antialising
            self.tk_image = ImageTk.PhotoImage(resized) # Converts the image into a format Tkinter can use
            self.label.config(image=self.tk_image) # Updates the label to show the image
            self.label.image = self.tk_image  # Keep a reference to the image to prevent it from being garbage collected!

    # Routine to increase the zooming level
    def zoom_in(self):
        self.zoom_level *= 1.2
        self.display_image()
    
    # Routine to decrease the zooming level
    def zoom_out(self):
        self.zoom_level /= 1.2
        self.display_image()

# Run the app
root = tk.Tk()
app = ImageViewer(root)
root.mainloop()
