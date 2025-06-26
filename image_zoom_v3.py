# Python code to create a window to display an image using tkinter.
# In this version scroll bars are added for too large images.
# The scroll bars can also be controlled by the arrow keys.

# Author Stefano Giani

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Class implementing the app
class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer with Zoom")
        self.root.geometry("512x512")
        self.root.resizable(False, False)


        self.zoom_level = 1.0
        self.original_image = None

        
        # Controls
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X)

        tk.Button(control_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(control_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        
        # Zoom controls with entry
        zoom_frame = tk.Frame(root)
        zoom_frame.pack(pady=5)

        tk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)

        self.zoom_entry = tk.Entry(zoom_frame, width=5)
        self.zoom_entry.insert(0, "100")  # Default 100%
        self.zoom_entry.pack(side=tk.LEFT)

        tk.Button(zoom_frame, text="+", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="-", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Set", command=self.set_zoom).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Reset", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)


        # Canvas with scrollbars
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scroll_y = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_y.grid(row=0, column=1, sticky="ns")

        self.scroll_x = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scroll_x.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        # Make the canvas expandable
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        # Bind + and - keys for zooming
        self.root.bind("<plus>", lambda event: self.zoom_in())
        self.root.bind("<minus>", lambda event: self.zoom_out())
        self.root.bind("<KeyPress-equal>", lambda event: self.zoom_in())  # For '+' without Shift
        self.root.bind("<KeyPress-minus>", lambda event: self.zoom_out())  # For '-' with Shift
        # Arrow keys to pan the image
        self.root.bind("<Left>", lambda event: self.canvas.xview_scroll(-1, "units"))
        self.root.bind("<Right>", lambda event: self.canvas.xview_scroll(1, "units"))
        self.root.bind("<Up>", lambda event: self.canvas.yview_scroll(-1, "units"))
        self.root.bind("<Down>", lambda event: self.canvas.yview_scroll(1, "units"))

        
        
        self.image_id = None

    def set_zoom(self):
        try:
            zoom_percent = float(self.zoom_entry.get())
            if zoom_percent <= 0:
                raise ValueError
            self.zoom_level = zoom_percent / 100.0
            self.display_image()
        except ValueError:
            print("Invalid zoom level. Please enter a positive number.")
            
    def reset_zoom(self):
        self.zoom_level = 1.0
        self.update_zoom_entry()
        self.display_image()

    # Routine to increase the zooming level
    def zoom_in(self):
        self.zoom_level *= 1.2
        self.update_zoom_entry()
        self.display_image()

    # Routine to decrease the zooming level
    def zoom_out(self):
        self.zoom_level /= 1.2
        self.update_zoom_entry()
        self.display_image()

    def update_zoom_entry(self):
        zoom_percent = int(self.zoom_level * 100)
        self.zoom_entry.delete(0, tk.END)
        self.zoom_entry.insert(0, str(zoom_percent))

        
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
            
            self.canvas.delete("all") # Clear previous image
            self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.canvas.config(scrollregion=self.canvas.bbox(self.image_id))


# Run the app
root = tk.Tk()
app = ImageViewer(root)
root.mainloop()
