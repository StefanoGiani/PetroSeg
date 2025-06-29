# Python app to crop an image.
#
# Features:
# - The image can be zoomed in and out.
# - Crop an image dragging a region with the mouse.
# - Ask to save the image, if not saved.

# Shortcuts:
# Ctrl+O Load image.
# Ctrl+Q Terminate the app.
# +, - Zoom level.
# Arrow keys Pan the image.

# TODO:

# Author Stefano Giani

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

STATUS_NONE = 0
STATUS_IMAGE_LOADED = 1

class ImageData:
    def __init__(self, rgb_image, flag_saved):
        self.rgb = rgb_image
        self.size = rgb_image.size
        self.saved = flag_saved

# Class implementing the app
class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer with Zoom")
        self.root.geometry("512x512")

        self.zoom_level = 1.0 # Level of Zoom
        self.image = None

        self.status = STATUS_NONE

        # Fields for cropping the image
        self.start_x = None
        self.start_y = None
        self.rect_id = None


        # Create the menu bar
        self.menu_bar = tk.Menu(root)


        # File menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open Image", accelerator="Ctrl+O", command=self.open_image)
        self.file_menu.add_command(label="Save Image", accelerator="Ctrl+S", command=self.save_image)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", accelerator="Ctrl+Q", command=self.on_closing)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.image_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.image_menu.add_command(label="Zoom in", accelerator="+", command=self.zoom_in)
        self.image_menu.add_command(label="Zoom out", accelerator="-", command=self.zoom_out)
        self.menu_bar.add_cascade(label="Image", menu=self.image_menu)

        # Attach the menu bar to the root window
        root.config(menu=self.menu_bar)


        self.file_menu.entryconfig("Save Image", state="disabled")
        self.image_menu.entryconfig("Zoom in", state="disabled")
        self.image_menu.entryconfig("Zoom out", state="disabled")
        
        # Controls
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X)

        self.status_label = tk.Label(root, text="Image size: N/A", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)
        
        # Zoom controls with entry
        zoom_frame = tk.Frame(root)
        zoom_frame.pack(pady=5)

        tk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)

        self.zoom_entry = tk.Entry(zoom_frame, width=5)
        self.zoom_entry.insert(0, "100")  # Default 100%
        self.zoom_entry.config(state="disabled")
        self.zoom_entry.pack(side=tk.LEFT)

        self.zoom_in_button = tk.Button(zoom_frame, text="+", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=2)
        self.zoom_in_button.config(state="disabled")
        self.zoom_out_button = tk.Button(zoom_frame, text="-", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT, padx=2)
        self.zoom_out_button.config(state="disabled")
        self.zoom_set_button = tk.Button(zoom_frame, text="Set", command=self.set_zoom)
        self.zoom_set_button.pack(side=tk.LEFT, padx=2)
        self.zoom_set_button.config(state="disabled")
        self.zoom_reset_button = tk.Button(zoom_frame, text="Reset", command=self.reset_zoom)
        self.zoom_reset_button.pack(side=tk.LEFT, padx=2)
        self.zoom_reset_button.config(state="disabled")


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
        # Menu file
        root.bind('<Control-o>', lambda event: self.open_image())
        root.bind('<Control-q>', lambda event: self.on_closing())

        # Events for cropping the image
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # Intercept message to close app
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
       
        self.image_id = None

    def set_zoom(self):
        if self.status == STATUS_IMAGE_LOADED:
            try:
                zoom_percent = float(self.zoom_entry.get())
                if zoom_percent <= 0:
                    raise ValueError
                self.zoom_level = zoom_percent / 100.0
                self.display_image()
            except ValueError:
                messagebox.showerror("Error", "Zoom percentage must be positive.")
                self.update_zoom_entry()
            
    def reset_zoom(self):
        if self.status == STATUS_IMAGE_LOADED:
            self.zoom_level = 1.0
            self.update_zoom_entry()
            self.display_image()

    # Routine to increase the zooming level
    def zoom_in(self):
        if self.status == STATUS_IMAGE_LOADED:
            self.zoom_level *= 1.2
            self.update_zoom_entry()
            self.display_image()

    # Routine to decrease the zooming level
    def zoom_out(self):
        if self.status == STATUS_IMAGE_LOADED:
            self.zoom_level /= 1.2
            self.update_zoom_entry()
            self.display_image()

    def update_zoom_entry(self):
        if self.status == STATUS_IMAGE_LOADED:
            zoom_percent = int(self.zoom_level * 100)
            self.zoom_entry.delete(0, tk.END)
            self.zoom_entry.insert(0, str(zoom_percent))

        
    # Routine to run in response to the button open image been pressed 
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.image = ImageData(Image.open(file_path), True)
            self.zoom_level = 1.0
            self.display_image()
            self.update_image_size_status(self.image.size)
            self.change_status(STATUS_IMAGE_LOADED)
    
    def update_image_size_status(self, size):
        self.status_label.config(text=f"Image size: {size[0]} x {size[1]} pixels")
            
    # Routine to display the image with the correct level of zooming
    def display_image(self):
        width, height = self.image.size
        new_size = (int(width * self.zoom_level), int(height * self.zoom_level))
        resized = self.image.rgb.resize(new_size, Image.LANCZOS) # Resize the image using antialising
        self.tk_image = ImageTk.PhotoImage(resized) # Converts the image into a format Tkinter can use
        
        self.canvas.delete("all") # Clear previous image
        self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(self.image_id))
        


    def on_mouse_press(self, event):
        if self.status == STATUS_IMAGE_LOADED:
            self.start_x = self.canvas.canvasx(event.x)
            self.start_y = self.canvas.canvasy(event.y)
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_mouse_drag(self, event):
        if self.status == STATUS_IMAGE_LOADED:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_mouse_release(self, event):
        if self.status == STATUS_IMAGE_LOADED:
            end_x = self.canvas.canvasx(event.x)
            end_y = self.canvas.canvasy(event.y)

            x1 = int(min(self.start_x, end_x) / self.zoom_level)
            y1 = int(min(self.start_y, end_y) / self.zoom_level)
            x2 = int(max(self.start_x, end_x) / self.zoom_level)
            y2 = int(max(self.start_y, end_y) / self.zoom_level)

            self.image = ImageData(self.image.rgb.crop((x1, y1, x2, y2)), False)
            self.zoom_level = 1.0
            self.display_image()
            self.update_image_size_status(self.image.size)
            self.file_menu.entryconfig("Save Image", state="normal")

    def save_image(self):
        if self.status == STATUS_IMAGE_LOADED:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                   filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                self.image.rgb.save(file_path)
                self.image.saved = True
                self.file_menu.entryconfig("Save Image", state="disabled")

    def change_status(self, new_status):
        if new_status == STATUS_IMAGE_LOADED:
            if self.status == STATUS_NONE:
                self.image_menu.entryconfig("Zoom in", state="normal")
                self.image_menu.entryconfig("Zoom out", state="normal")
                self.zoom_entry.config(state="normal")
                self.zoom_in_button.config(state="normal")
                self.zoom_out_button.config(state="normal")
                self.zoom_set_button.config(state="normal")
                self.zoom_reset_button.config(state="normal")
                self.status = STATUS_IMAGE_LOADED
            else:
                messagebox.showerror("Error", "No transition for the status")

        else:
            messagebox.showerror("Error", "Status unknown")


    def on_closing(self):
        if self.status == STATUS_IMAGE_LOADED:
            if not self.image.saved:
                result = messagebox.askyesnocancel("Exit", "Do you want to save the image before exiting?")
                if result is True:
                    self.save_image()
                    if self.image.saved:
                        self.root.destroy()
                elif result is False:
                    self.root.destroy()
            else:
                self.root.destroy()
        else:
            self.root.destroy()



# Run the app
root = tk.Tk()
app = ImageViewer(root)
root.mainloop()
