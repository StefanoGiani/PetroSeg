import tkinter as tk
from tkinter import ttk, messagebox

class HSVDialog(tk.Toplevel):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.title("Set HSV Parameters")
        self.geometry("300x320")
        self.resizable(False, False)
        self.callback = callback

        self.params = {
            'H': {'value': 0, 'min': 0, 'max': 360},
            'S': {'value': 0, 'min': 0, 'max': 100},
            'V': {'value': 0, 'min': 0, 'max': 100}
        }

        self.entries = {}
        self.buttons = {}
        self.mode = tk.StringVar(value="HV")

        self.create_widgets()

    def create_widgets(self):
        # Radio buttons
        mode_frame = ttk.LabelFrame(self, text="Mode")
        mode_frame.pack(pady=10)

        hv_radio = ttk.Radiobutton(mode_frame, text="H and V", variable=self.mode, value="HV", command=self.update_mode)
        s_radio = ttk.Radiobutton(mode_frame, text="S only", variable=self.mode, value="S", command=self.update_mode)
        hv_radio.grid(row=0, column=0, padx=10)
        s_radio.grid(row=0, column=1, padx=10)

        # Parameter controls
        for param in self.params:
            frame = ttk.Frame(self)
            frame.pack(pady=5)

            label = ttk.Label(frame, text=f"{param}:")
            label.grid(row=0, column=0, padx=5)

            minus_btn = ttk.Button(frame, text="-", width=3,
                                   command=lambda p=param: self.update_value(p, -1))
            minus_btn.grid(row=0, column=1)

            entry = ttk.Entry(frame, width=5, justify='center')
            entry.insert(0, str(self.params[param]['value']))
            entry.grid(row=0, column=2)

            plus_btn = ttk.Button(frame, text="+", width=3,
                                  command=lambda p=param: self.update_value(p, 1))
            plus_btn.grid(row=0, column=3)

            self.entries[param] = entry
            self.buttons[param] = (minus_btn, plus_btn)

        self.update_mode()

        # Submit and Cancel buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=15)

        submit_btn = ttk.Button(btn_frame, text="Submit", command=self.submit)
        submit_btn.grid(row=0, column=0, padx=10)

        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.cancel)
        cancel_btn.grid(row=0, column=1, padx=10)

    def update_mode(self):
        mode = self.mode.get()
        for param in self.params:
            active = (mode == "HV" and param in ("H", "V")) or (mode == "S" and param == "S")
            state = "normal" if active else "disabled"
            self.entries[param].config(state=state)
            for btn in self.buttons[param]:
                btn.config(state=state)

    def update_value(self, param, delta):
        try:
            current = int(self.entries[param].get())
        except ValueError:
            current = self.params[param]['min']
        new_val = max(self.params[param]['min'], min(self.params[param]['max'], current + delta))
        self.entries[param].delete(0, tk.END)
        self.entries[param].insert(0, str(new_val))

    def submit(self):
        values = {}
        for param in self.params:
            try:
                val = int(self.entries[param].get())
                if self.params[param]['min'] <= val <= self.params[param]['max']:
                    values[param] = val
                else:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid value for {param}")
                return
        selected_mode = self.mode.get()
        self.callback(values, selected_mode)  # Pass both values and mode
        self.destroy()


    def cancel(self):
        self.callback(None, None)
        self.destroy()




class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Main App")
        self.geometry("300x150")

        open_btn = ttk.Button(self, text="Open HSV Dialog", command=self.open_dialog)
        open_btn.pack(pady=20)

        self.result_label = ttk.Label(self, text="HSV Values: Not set")
        self.result_label.pack()

    def open_dialog(self):
        HSVDialog(self, self.receive_values)

    def receive_values(self, values, mode):
        if values is None:
            self.result_label.config(text="HSV Values: Cancelled")
            print("Dialog was cancelled.")
        else:
            self.result_label.config(
                text=f"HSV Values: H={values['H']}, S={values['S']}, V={values['V']} | Mode: {mode}"
            )
            print("Received HSV values:", values, "Mode:", mode)




if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
