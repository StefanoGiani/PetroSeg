# Python code to create a window to display an image using tkinter.

# Author Stefano Giani


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Routine to run in response to the button open image been pressed 
def open_image():
    # Open dialog to load an image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if file_path:
        img = Image.open(file_path)
        img = img.resize((400, 400))  # Resize for display
        img_tk = ImageTk.PhotoImage(img) # Converts the image into a format Tkinter can use
        label.config(image=img_tk) # Updates the label to show the image
        label.image = img_tk  # Keep a reference to the image to prevent it from being garbage collected!

# Create main window
root = tk.Tk()
root.title("Image Viewer")

# Create a button to open image
btn = tk.Button(root, text="Open Image", command=open_image) # When clicked, it calls the open_image() function
btn.pack(pady=10) # Adds vertical padding around the button

# Label to display the image
label = tk.Label(root) # A Label widget is created to hold the image
label.pack() # Place the label below the button because the button has been packed before

# Start the GUI loop
root.mainloop()
