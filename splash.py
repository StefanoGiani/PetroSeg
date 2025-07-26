import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

def show_splash(root, image_path, duration=3000):
    splash = tk.Toplevel(root)
    splash.overrideredirect(True)  # Remove window decorations

    # Load and display image
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)
    img_label = tk.Label(splash, image=photo)
    img_label.image = photo  # Keep a reference
    img_label.pack()

    # Add text below the image
    text_frame = tk.Frame(splash, bg="white")
    text_frame.pack(fill="both", expand=True)
    tk.Label(text_frame, text="Authors: Stefano Giani, et al.", bg="white", font=("Arial", 12)).pack(pady=(10, 0))
    tk.Label(text_frame, text="License: MIT", bg="white", font=("Arial", 10)).pack()

    # Center the splash screen
    splash.update_idletasks()
    width = splash.winfo_width()
    height = splash.winfo_height()
    x = (splash.winfo_screenwidth() // 2) - (width // 2)
    y = (splash.winfo_screenheight() // 2) - (height // 2)
    splash.geometry(f"{width}x{height}+{x}+{y}")

    # Hide splash after duration and show main window
    root.withdraw()
    root.after(duration, lambda: (splash.destroy(), root.deiconify()))

def main_app():
    root = tk.Tk()
    root.title("Main Application")
    root.geometry("400x300")

    show_splash(root, "dice.jpeg", duration=3000)

    # Main app content
    ttk.Label(root, text="Welcome to the Main App!").pack(pady=100)

    root.mainloop()

if __name__ == "__main__":
    main_app()
