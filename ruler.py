import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import math

class VirtualRulerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Ruler")
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        self.image = None
        self.tk_image = None
        self.start_point = None
        self.temp_line = None
        self.final_line = None
        self.temp_text = None
        self.markers = []

        self.unit = "px"
        self.conversion_factors = {"px": 1.0, "cm": None, "mm": None, "in": None}

        # Menu
        menu = tk.Menu(root)
        root.config(menu=menu)
        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Calibrate", command=self.open_calibration_dialog)

        # Bindings
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_drag)
        self.root.bind("<Escape>", self.reset_ruler)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.reset_ruler()

    def on_click(self, event):
        if self.start_point is None:
            self.reset_ruler()
            self.start_point = (event.x, event.y)
            marker = self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red")
            self.markers.append(marker)
        else:
            end_point = (event.x, event.y)
            if self.temp_line:
                self.canvas.delete(self.temp_line)
            if self.temp_text:
                self.canvas.delete(self.temp_text)
            self.final_line = self.canvas.create_line(self.start_point[0], self.start_point[1],
                                                      end_point[0], end_point[1], fill="blue", width=2)
            marker = self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red")
            self.markers.append(marker)
            distance_px = math.hypot(end_point[0] - self.start_point[0], end_point[1] - self.start_point[1])
            distance = distance_px * self.conversion_factors.get(self.unit, 1.0)
            self.temp_text = self.canvas.create_text((self.start_point[0] + end_point[0]) // 2,
                                                     (self.start_point[1] + end_point[1]) // 2,
                                                     text=f"{distance:.2f} {self.unit}", fill="black")
            self.start_point = None

    def on_drag(self, event):
        if self.start_point:
            if self.temp_line:
                self.canvas.delete(self.temp_line)
            if self.temp_text:
                self.canvas.delete(self.temp_text)
            self.temp_line = self.canvas.create_line(self.start_point[0], self.start_point[1],
                                                     event.x, event.y, fill="gray", dash=(4, 2))
            distance_px = math.hypot(event.x - self.start_point[0], event.y - self.start_point[1])
            distance = distance_px * self.conversion_factors.get(self.unit, 1.0)
            self.temp_text = self.canvas.create_text((self.start_point[0] + event.x) // 2,
                                                     (self.start_point[1] + event.y) // 2,
                                                     text=f"{distance:.2f} {self.unit}", fill="gray")

    def reset_ruler(self, event=None):
        self.start_point = None
        if self.temp_line:
            self.canvas.delete(self.temp_line)
            self.temp_line = None
        if self.final_line:
            self.canvas.delete(self.final_line)
            self.final_line = None
        if self.temp_text:
            self.canvas.delete(self.temp_text)
            self.temp_text = None
        for marker in self.markers:
            self.canvas.delete(marker)
        self.markers.clear()

    def open_calibration_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Calibrate Ruler")

        tk.Label(dialog, text="Distance in pixels:").grid(row=0, column=0, padx=5, pady=5)
        pixel_entry = tk.Entry(dialog)
        pixel_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(dialog, text="Real-world distance:").grid(row=1, column=0, padx=5, pady=5)
        real_entry = tk.Entry(dialog)
        real_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(dialog, text="Unit:").grid(row=1, column=2, padx=5, pady=5)
        unit_var = tk.StringVar(value="cm")
        unit_menu = ttk.Combobox(dialog, textvariable=unit_var, values=["cm", "mm", "in"], state="readonly", width=5)
        unit_menu.grid(row=1, column=3, padx=5, pady=5)

        tk.Label(dialog, text="Display unit:").grid(row=2, column=0, padx=5, pady=5)
        display_unit_var = tk.StringVar(value=self.unit)
        for i, u in enumerate(["px", "cm", "mm", "in"]):
            tk.Radiobutton(dialog, text=u, variable=display_unit_var, value=u).grid(row=2, column=1+i, padx=2, pady=5)

        def apply_calibration():
            try:
                px = float(pixel_entry.get())
                real = float(real_entry.get())
                unit = unit_var.get()
                factor = real / px
                self.conversion_factors["px"] = 1.0
                self.conversion_factors["cm"] = factor if unit == "cm" else factor / 10 if unit == "mm" else factor * 2.54
                self.conversion_factors["mm"] = self.conversion_factors["cm"] * 10
                self.conversion_factors["in"] = self.conversion_factors["cm"] / 2.54
                self.unit = display_unit_var.get()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers.")

        tk.Button(dialog, text="Apply", command=apply_calibration).grid(row=3, column=0, columnspan=4, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualRulerApp(root)
    root.mainloop()
