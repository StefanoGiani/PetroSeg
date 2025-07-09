# Python code to create a window to display an image using tkinter.
# In this version clicking on the image, the coordinated are diplayed at the bottom of the windoe.

# Author Stefano Giani

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

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
        tk.Button(control_frame, text="Detect Inclusions", command=self.detect_inclusions).pack(side=tk.LEFT, padx=5)
        
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

    def display_coordinates(self, event):
        # Convert canvas coordinates to image coordinates considering zoom
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        image_x = int(canvas_x / self.zoom_level)
        image_y = int(canvas_y / self.zoom_level)
        self.coord_label.config(text=f"Clicked at: ({image_x}, {image_y})")
        
    def detect_inclusions(self):
        
        
        min_area = 1 # Minimum area threshold
        max_area = 1000 # Maximum area threshold

        # Set ratio threshold
        ratio_threshold = 0.8
        
        # Convert image to greyscale
        image = np.array(self.original_image.convert('RGB'))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        H_min = np.min(hsv_image[:,:,0])
        H_max = np.max(hsv_image[:,:,0])
        S_min = np.min(hsv_image[:,:,1])
        S_max = np.max(hsv_image[:,:,1])
        V_min = np.min(hsv_image[:,:,2])
        V_max = np.max(hsv_image[:,:,2])
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray image',gray_image)
        #cv2.waitKey(0)
        # Apply edge detection
        # Fine-tuning the thresholds in the cv2.Canny() function can help. 
        # Lower thresholds might detect more edges, while higher thresholds might reduce noise.
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
        #cv2.imshow('edges',edges)
        #cv2.waitKey(0)
        # Find contours
        # Experiment with different methods for contour retrieval (cv2.RETR_EXTERNAL, cv2.RETR_TREE, etc.) 
        # and approximation (cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE).
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create masks for color detection
        lower_bound = np.array([H_min, S_min, V_min])
        upper_bound = np.array([H_max, S_max, V_max])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        #cv2.imshow('mask',mask)
        #cv2.waitKey(0)
        
        # Morphological Operations
        # Use morphological operations like cv2.dilate() and cv2.erode() to refine the masks and contours.
        #kernel = np.ones((5, 5), np.uint8)
        #dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        #eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)

        # Combine Contours and color masks
        filtered_contours = []
        for contour in contours:
            
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            if min_area <= area <= max_area:
                if hull_area > 0:  # Avoid division by zero
                    ratio = area / hull_area
                    if ratio >= ratio_threshold:
                        filtered_contours.append(contour)
        inclusion_count = 0
        mask_incusions = np.zeros_like(gray_image)
        for contour in filtered_contours:
            mask_contour = np.zeros_like(gray_image)
            cv2.drawContours(mask_contour, [contour], -1, 255, thickness=cv2.FILLED)
            inclusion_mask = cv2.bitwise_and(mask, mask, mask=mask_contour)
            #cv2.imshow('inclusion',inclusion_mask)
            #cv2.waitKey(0)
            inclusions = cv2.findNonZero(inclusion_mask)
            # add masks
            mask_incusions += inclusion_mask
            #cv2.imshow('mask_incusions',mask_incusions)
            #cv2.waitKey(0)
            

            if inclusions is not None:
                inclusion_count += 1
        
        mask_incusions = mask_incusions.clip(0, 255).astype("uint8")
        self.original_image = Image.fromarray(mask_incusions)
        self.display_image()
        print(inclusion_count)

            
# Run the app
root = tk.Tk()
app = ImageViewer(root)
root.mainloop()
