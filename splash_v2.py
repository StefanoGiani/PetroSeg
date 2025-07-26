import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def show_splash(root, image_path, duration=3000):
    splash = tk.Toplevel(root)
    splash.overrideredirect(True)

    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)
    img_label = tk.Label(splash, image=photo)
    img_label.image = photo
    img_label.pack()

    text_frame = tk.Frame(splash, bg="white")
    text_frame.pack(fill="both", expand=True)
    tk.Label(text_frame, text="Authors: Stefano Giani, et al.", bg="white", font=("Arial", 12)).pack(pady=(10, 0))
    tk.Label(text_frame, text="License: MIT", bg="white", font=("Arial", 10)).pack()

    splash.update_idletasks()
    width = splash.winfo_width()
    height = splash.winfo_height()
    x = (splash.winfo_screenwidth() // 2) - (width // 2)
    y = (splash.winfo_screenheight() // 2) - (height // 2)
    splash.geometry(f"{width}x{height}+{x}+{y}")

    root.withdraw()
    root.after(duration, lambda: (splash.destroy(), root.deiconify()))

def show_about_window(root, image_path):
    about = tk.Toplevel(root)
    about.title("About")
    about.resizable(False, False)

    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)
    img_label = tk.Label(about, image=photo)
    img_label.image = photo
    img_label.pack()

    text_frame = tk.Frame(about, bg="white")
    text_frame.pack(fill="both", expand=True)
    tk.Label(text_frame, text="Authors: Stefano Giani, et al.", bg="white", font=("Arial", 12)).pack(pady=(10, 0))
    tk.Label(text_frame, text="License: MIT", bg="white", font=("Arial", 10)).pack()

    ttk.Button(about, text="Close", command=about.destroy).pack(pady=10)

    about.update_idletasks()
    width = about.winfo_width()
    height = about.winfo_height()
    x = (about.winfo_screenwidth() // 2) - (width // 2)
    y = (about.winfo_screenheight() // 2) - (height // 2)
    about.geometry(f"{width}x{height}+{x}+{y}")

def main_app():
    root = tk.Tk()
    root.title("Main Application")
    root.geometry("400x300")

    # Menu bar with About
    menubar = tk.Menu(root)
    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_command(label="About", command=lambda: show_about_window(root, "dice.jpeg"))
    menubar.add_cascade(label="Help", menu=helpmenu)
    root.config(menu=menubar)

    show_splash(root, "dice.jpeg", duration=3000)

    ttk.Label(root, text="Welcome to the Main App!").pack(pady=100)

    root.mainloop()

if __name__ == "__main__":
    main_app()
