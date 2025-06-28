# Python app to segment inclusions in images
#
# Features:
# - Possibility to laod two images. This feature will let the user load an image and a mask.
# - Possibility to zoom in and out.
# - Clicking anywhere on an image, will display the coordinates of the click at the bottom of the window.

# TODO:
# - Substitute the second image with a mask.

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


        self.zoom_level = 1.0
        self.image1 = None
        self.image2 = None
        self.image1_size = (0, 0)
        self.image2_size = (0, 0)
        self.current_img = 0


        # Create the menu bar
        menu_bar = tk.Menu(root)


        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image 1", accelerator="Ctrl+O", command=self.open_image1)
        file_menu.add_command(label="Open Image 2", command=self.open_image2)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", accelerator="Ctrl+Q", command=self.exit_app)
        menu_bar.add_cascade(label="File", menu=file_menu)


        # Attach the menu bar to the root window
        root.config(menu=menu_bar)


        # Bind keyboard shortcuts
        root.bind('<Control-o>', lambda event: self.open_image1())
        root.bind('<Control-q>', lambda event: self.exit_app())



        
        # Controls
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X)

        self.load_button1 = tk.Button(control_frame, text="Load Image 1", command=self.open_image1)
        self.load_button1.pack(side=tk.LEFT, padx=5)

        self.load_button2 = tk.Button(control_frame, text="Load Image 2", command=self.open_image2)
        self.load_button2.pack(side=tk.LEFT, padx=5)

        self.show_button1 = tk.Button(control_frame, text="Image 1", command=self.show_image1, state=tk.DISABLED)
        self.show_button1.pack(side=tk.LEFT, padx=5)

        self.show_button2 = tk.Button(control_frame, text="Image 2", command=self.show_image2, state=tk.DISABLED)
        self.show_button2.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(root, text="Image size: N/A", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)
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
        self.root.bind("<KP_Add>", lambda event: self.zoom_in())  # For '+' without Shift
        self.root.bind("<KP_Subtract>", lambda event: self.zoom_out())  # For '-' with Shift
        # Arrow keys to pan the image
        self.root.bind("<Left>", lambda event: self.canvas.xview_scroll(-1, "units"))
        self.root.bind("<Right>", lambda event: self.canvas.xview_scroll(1, "units"))
        self.root.bind("<Up>", lambda event: self.canvas.yview_scroll(-1, "units"))
        self.root.bind("<Down>", lambda event: self.canvas.yview_scroll(1, "units"))
        
        # Bottom frame to show coordinates
        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(fill=tk.X)
        self.coord_label = tk.Label(self.bottom_frame, text="Clicked at: ", anchor="w")
        self.coord_label.pack(fill="x")

        # Bind mouse click to coordinate display
        self.canvas.bind("<Button-1>", self.display_coordinates)
       
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

    def display_coordinates(self, event):
        # Convert canvas coordinates to image coordinates considering zoom
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        image_x = int(canvas_x / self.zoom_level)
        image_y = int(canvas_y / self.zoom_level)
        self.coord_label.config(text=f"Clicked at: ({image_x}, {image_y})")

        
    # Routine to run in response to the button open image been pressed 
    def open_image1(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.image1 = Image.open(file_path)
            self.zoom_level = 1.0
            self.image1_size = self.image1.size
            #self.image1 = ImageTk.PhotoImage(img)
            self.current_img = 1
            self.display_image()
            self.update_status(self.image1_size)
            self.check_images_loaded()

    def open_image2(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.image2 = Image.open(file_path)
            self.zoom_level = 1.0
            self.image2_size = self.image2.size
            #self.image2 = ImageTk.PhotoImage(img)
            self.current_img = 2
            self.display_image()
            self.update_status(self.image2_size)
            self.check_images_loaded()

    def check_images_loaded(self):
        if self.image1 and self.image2:
            self.show_button1.config(state=tk.NORMAL)
            self.show_button2.config(state=tk.NORMAL)
    
    def update_status(self, size):
        self.status_label.config(text=f"Image size: {size[0]} x {size[1]} pixels")
            
    # Routine to display the image with the correct level of zooming
    def display_image(self):
        if self.current_img == 1:
            width, height = self.image1_size
            new_size = (int(width * self.zoom_level), int(height * self.zoom_level))
            resized = self.image1.resize(new_size, Image.LANCZOS) # Resize the image using antialising
            self.tk_image = ImageTk.PhotoImage(resized) # Converts the image into a format Tkinter can use
            
            self.canvas.delete("all") # Clear previous image
            self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.canvas.config(scrollregion=self.canvas.bbox(self.image_id))
        elif self.current_img == 2:
            width, height = self.image2_size
            new_size = (int(width * self.zoom_level), int(height * self.zoom_level))
            resized = self.image2.resize(new_size, Image.LANCZOS) # Resize the image using antialising
            self.tk_image = ImageTk.PhotoImage(resized) # Converts the image into a format Tkinter can use
            
            self.canvas.delete("all") # Clear previous image
            self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.canvas.config(scrollregion=self.canvas.bbox(self.image_id))

    def show_image1(self):
        if self.image1:
            self.current_img = 1
            self.display_image()
            self.update_status(self.image1_size)

    def show_image2(self):
        if self.image2:
            self.current_img = 2
            self.display_image()
            self.update_status(self.image2_size)

    
    def exit_app(self):
        root.quit()



# Run the app
root = tk.Tk()
app = ImageViewer(root)
root.mainloop()
