# Python app to crop an image.
#
# Features:
# - The image can be zoomed in and out.
# - Crop an image dragging a region with the mouse.
# - Undo and redo.
# - Added mechanism for tools.
# - Added message label in the bottom status bar.
# - Addition of a mask to mark regions. The mask is RGB to allow for more colors and clearer results.
# - Side panel to report color values and coordinates of the cursor.
# - Tools to pick and unpick colors to add and remove regions in the mask.

# Shortcuts:
# Ctrl+N New project and open an image.
# Ctrl+S Save project.
# Ctrl+O Load project.
# Ctrl+Q Terminate the app.
# +, - Zoom level.
# Arrow keys Pan the image.
# Ctrl+Z Undo.
# Ctrl+Y Redo.
# Esc Deselect current tool.
# Right Mouse Buttom Deselect current tool.
# 1, 2, 3 Switch from image, to mask and to blend of image and mask.

# TODO:
# - Masking background.

# Author Stefano Giani

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pickle
import cv2
import numpy as np

# Stati of the app:
STATUS_NONE = 0 # Just started 
STATUS_IMAGE_LOADED = 1 # An image is loaded

# Stati of the tools
STATUS_TOOL_NONE = 0 # No tool selected.
STATUS_TOOL_CROP = 1 # Crop selected
STATUS_TOOL_PICK_COLOR = 2 # Pick color selected
STATUS_TOOL_UNPICK_COLOR = 3 # Unpick color selected

# Stati for the displayied image
STATUS_DISPLAY_NONE = 0
STATUS_DISPLAY_IMAGE = 1
STATUS_DISPLAY_MASK = 2
STATUS_DISPLAY_MIX = 3

# List of colors for the mask to mark differnet regions
COLOR_MASK_NONE = (0,0,0)
COLOR_MASK_BACKGROUND = (255,255,255)
COLOR_MASK_MATRIX = (255,0,0)
COLOR_MASK_INCLUSION = (0,255,0)

# Class to contain the data for the projecy
class ProjectData:
    def __init__(self, project_path = None, image_path = None ):
        """Constructor.
        It can be called with no parameters to create an empty project, 
        or it can be called passing either a path of an image file or a
        path of a project file.

        Parameters
        ----------
        project_path : string
            Path to the file containing the project to open.
        image_path : string
            Path to the file containing the image to open.
        """
        if image_path is not None:
        
            self.rgb = Image.open(image_path).convert('RGB')
            img_tmp = np.array(self.rgb)
            self.hsv = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2HSV)
            self.size = self.rgb.size
            # cv2 dimensions are inverted compared to PIL
            self.mask = np.zeros((self.size[1], self.size[0],3), dtype=np.uint8)
            self.saved = False
            self.stack_actions = []
            self.stack_actions.append({"action": "LoadImage(" + image_path + ")", "saved": self.saved, "size": self.size, "rgb": self.rgb.copy(), "hsv": self.hsv.copy(), 
                                       "mask":self.mask.copy()})
            self.cursor_stack_action = 0
            self.project_file = None
        elif project_path is not None:
            self.load(project_path)
        else:
            self.rgb = None
            self.hsv = None
            self.mask = None
            self.size = None
            self.saved = True
            self.stack_actions = []
            self.cursor_stack_action = 0
            self.project_file = None

    def save(self, project_path):
        """Routine to save the project to file.

        Parameters
        ----------
        project_path : string
           Path where to save the project.
        """

        with open(project_path, "wb") as f:
            state= {}
            state['rgb'] = self.rgb
            state['hsv'] = self.hsv
            state['size'] = self.size
            state['stack_actions'] = self.stack_actions
            state['cursor_stack_action'] = self.cursor_stack_action
            state['project_file'] = project_path
            state['mask'] = self.mask
            self.project_file = project_path
            pickle.dump(state, f)
        self.saved = True

    def load(self, project_path):
        """Routine to load a project from disk.

        Parameters
        ----------
        project_path : string
            _description_
        """

        with open(project_path, "rb") as f:
            state = pickle.load(f)
            self.rgb = state['rgb'] 
            self.hsv = state['hsv'] 
            self.size = state['size'] 
            self.stack_actions = state['stack_actions']
            self.cursor_stack_action = state['cursor_stack_action']
            self.project_file = state['project_file']
            self.mask = state['mask']
        self.saved = True


    def select_color_hsv(self, lower, upper, color):
        """Select a region in the mask based on range in HSV colors.
        
        Parameters
        ----------
        lower : tuple
            Three uint8 values to determine the lower value for the range in HSV.
        upper : tuple
            Three uint8 values to determine the upper value for the range in HSV.
        color: tuple
            Three uint8 values to determine the RGB color to use to mark the region on the mask."""
        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        mask = cv2.inRange(self.hsv, lower, upper)
        # Apply the color where mask == 255
        self.mask[mask == 255] = color
        self.saved = False
        self.stack_actions.append({"action": "SelectColorHSV(" + str(lower) + "," + str(upper)  + "," + str(color) + ")", "saved": self.saved, "mask": self.mask.copy()})
        


    def truncate_stack(self):
        """Truncate the stack of actions.

        This can be used when an action is performed on the data and the remaining redo actions are no longer applicable.
        """
        if self.cursor_stack_action > 0:
            del self.stack_actions[self.cursor_stack_action+1:]

    def crop_image(self, x1, y1, x2, y2):
        """Routine to crop an image using coordinates of opposite vertices of a rectangle.

        Parameters
        ----------
        x1 : integer
            Coordinate.
        y1 : integer
            Coordinate.
        x2 : integer
            Coordinate.
        y2 : integer
            Coordinate.
        """

        # Make sure the coordinates make sense
        if x1<0:
            x1 = 0
        if y1<0:
            y1 = 0
        if x2>=self.size[0]:
            x2 = self.size[0]-1
        if y2>=self.size[1]:
            y2 = self.size[1]-1

        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        self.rgb = self.rgb.crop((x1, y1, x2, y2))
        img_tmp = np.array(self.rgb)
        self.hsv = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2HSV)
        self.mask = self.mask[y1:y2, x1:x2, :]
        self.size = self.rgb.size
        self.saved = False
        self.stack_actions.append({"action": "CropImage(" + str(x1) + "," + str(y1) + "," +  str(x2) + "," +  str(y2) + ")", "saved": self.saved, 
                                   "size": self.size, "rgb": self.rgb.copy(), "hsv": self.hsv.copy(), "mask": self.mask.copy()})
        

    def undo(self):
        """Undo the last action.

        The first action cannot be undo.

        """
        if self.cursor_stack_action > 0:
            self.cursor_stack_action -= 1
            if "saved" in self.stack_actions[self.cursor_stack_action].keys():
                self.saved = self.stack_actions[self.cursor_stack_action]["saved"]
            if "size" in self.stack_actions[self.cursor_stack_action].keys():
                self.size = self.stack_actions[self.cursor_stack_action]["size"]
            if "rgb" in self.stack_actions[self.cursor_stack_action].keys():
                self.rgb = self.stack_actions[self.cursor_stack_action]["rgb"]
            if "hsv" in self.stack_actions[self.cursor_stack_action].keys():
                self.hsv = self.stack_actions[self.cursor_stack_action]["hsv"]
            if "mask" in self.stack_actions[self.cursor_stack_action].keys():
                self.mask = self.stack_actions[self.cursor_stack_action]["mask"]
            
        
    def redo(self):
        """Redo the next action in the stack.
        """
        if self.cursor_stack_action < len(self.stack_actions)-1:
            self.cursor_stack_action += 1
            if "saved" in self.stack_actions[self.cursor_stack_action].keys():
                self.saved = self.stack_actions[self.cursor_stack_action]["saved"]
            if "size" in self.stack_actions[self.cursor_stack_action].keys():
                self.size = self.stack_actions[self.cursor_stack_action]["size"]
            if "rgb" in self.stack_actions[self.cursor_stack_action].keys():
                self.rgb = self.stack_actions[self.cursor_stack_action]["rgb"]
            if "hsv" in self.stack_actions[self.cursor_stack_action].keys():
                self.hsv = self.stack_actions[self.cursor_stack_action]["hsv"]
            if "mask" in self.stack_actions[self.cursor_stack_action].keys():
                self.mask = self.stack_actions[self.cursor_stack_action]["mask"]

    def is_undo(self):
        """Routine to check if there are actions in the stack that can be undo.

        The first action cannot be undo.

        Returns
        -------
        logical
            Status
        """
        if self.cursor_stack_action > 0:
            return True
        else:
            return False
        
    def is_redo(self):
        """Routine to check if there are actions in the stack that can be redo.

        Returns
        -------
        logical
            Status
        """
        if self.cursor_stack_action < len(self.stack_actions)-1:
            return True
        else:
            return False

# Class for creating the dialog to set up the parameters for the tool pick color
class PickColorDialog(tk.Toplevel):
    def __init__(self, parent, callback, initial_values=None):
        """Constructor.

        Parameters
        ----------
        parent : TK class
            Parent object.
        callback : function
            Routine to call to return parameter values to the caller.
        initial_values : dictionary, optional
            Initialisation value for the parameters, by default None
        """
        super().__init__(parent)
        self.title("Pick Color Dialogue")
        self.geometry("300x350")
        self.resizable(False, False)
        self.callback = callback

        # Parameter names and range
        self.params = {
            'H': {'value': 0, 'min': 0, 'max': 179},
            'S': {'value': 0, 'min': 0, 'max': 255},
            'V': {'value': 0, 'min': 0, 'max': 255},
            'R': {'value': 0, 'min': 0, 'max': 255},
            'G': {'value': 0, 'min': 0, 'max': 255},
            'B': {'value': 0, 'min': 0, 'max': 255}
        }

        # Parameter symbols
        self.symbol_label = {
            'H': "±ΔH",
            'S': "±ΔS",
            'V': "±ΔV",
            'R': "±ΔR",
            'G': "±ΔG",
            'B': "±ΔB"
        }
        
        self.mode = tk.StringVar(value='HSV')
        # Override with initial values if provided
        if initial_values:
            for key in self.params:
                if key in initial_values:
                    self.params[key]['value'] = initial_values[key]
            if 'mode' in initial_values:
                self.mode = tk.StringVar(value=initial_values['mode'])

        self.entries = {}
        self.buttons = {}
        
        self.create_widgets()

    def create_widgets(self):
        """Routine to create the dialogue.
        """
        # Radio buttons
        mode_frame = ttk.LabelFrame(self, text="Mode")
        mode_frame.pack(pady=10)

        hsv_radio = ttk.Radiobutton(mode_frame, text="HSV", variable=self.mode, value="HSV", command=self.update_mode)
        rgb_radio = ttk.Radiobutton(mode_frame, text="RGB", variable=self.mode, value="RGB", command=self.update_mode)
        hsv_radio.grid(row=0, column=0, padx=10)
        rgb_radio.grid(row=0, column=1, padx=10)

        # Parameter controls
        for param in ['H', 'S', 'V']:
            frame = ttk.Frame(self)
            frame.pack(pady=5)

            label = ttk.Label(frame, text=self.symbol_label[param])

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
            
        # Separator between HSV and RGB
        ttk.Separator(self, orient='horizontal').pack(fill='x', pady=10)
        
        # Parameter controls
        for param in ['R', 'G', 'B']:
            frame = ttk.Frame(self)
            frame.pack(pady=5)

            label = ttk.Label(frame, text=self.symbol_label[param])

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

        self.update_mode()  # Set initial state
        
        # Submit and Cancel buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=15)

        submit_btn = ttk.Button(btn_frame, text="Submit", command=self.submit)
        submit_btn.grid(row=0, column=0, padx=10)

        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.cancel)
        cancel_btn.grid(row=0, column=1, padx=10)

    def update_mode(self):
        """Routine to select between HSV and RGB.
        """
        mode = self.mode.get()
        for param in self.params:
            active = (mode == "HSV" and param in ("H", "S", "V")) or (mode == "RGB" and param in ("R", "G", "B"))
            state = "normal" if active else "disabled"
            self.entries[param].config(state=state)
            for btn in self.buttons[param]:
                btn.config(state=state)

    def update_value(self, param, delta):
        """Routine to update the value of the parameters using the buttons."""
        try:
            current = int(self.entries[param].get())
        except ValueError:
            current = self.params[param]['min']
        new_val = max(self.params[param]['min'], min(self.params[param]['max'], current + delta))
        self.entries[param].delete(0, tk.END)
        self.entries[param].insert(0, str(new_val))

    def submit(self):
        """Routine to submit the changes."""
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
        """Routine to cancel the dialogue."""
        self.callback(None, None)
        self.destroy()
        
# Class implementing the app
class ImageViewer:
    def __init__(self, root):
        """Constructor for the app.

        Parameters
        ----------
        root : Tkinter root
            Root
        """
        self.root = root
        self.root.title("Image Viewer with Zoom")
        self.root.geometry("512x512")

        self.zoom_level = 1.0 # Level of Zoom
        self.project = ProjectData()

        # Init initial status app
        self.status = STATUS_NONE

        # Init initial status tools
        self.status_tool = STATUS_TOOL_NONE

        # Fields for cropping the image
        self.start_x = None
        self.start_y = None
        self.rect_id = None

        # Variable for the display
        self.display_status = STATUS_DISPLAY_NONE
        
        # Selected color to use on the mask
        self.current_mask_color = COLOR_MASK_NONE
        
        # Tolerance on HSV and RGB values for the tool pick color
        self.pick_color_params = {
            'H': 0,
            'S': 0,
            'V': 0,
            'R': 0,
            'G': 0,
            'B': 0,
            "mode": 'HSV'
        }

        # Create the menu bar
        self.menu_bar = tk.Menu(root)
        # File menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="New Project...", accelerator="Ctrl+N", command=self.new_project)
        self.file_menu.add_command(label="Load Project...", accelerator="Ctrl+O", command=self.load_project)
        self.file_menu.add_command(label="Save Project", accelerator="Ctrl+S", command=self.save_project)
        self.file_menu.add_command(label="Save Project As...", command=self.save_as_project)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", accelerator="Ctrl+Q", command=self.on_closing)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        # Edit Menu
        self.edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.edit_menu.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        self.edit_menu.add_command(label="Redo", accelerator="Ctrl+Y", command=self.redo)
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Zoom in", accelerator="+", command=self.zoom_in)
        self.edit_menu.add_command(label="Zoom out", accelerator="-", command=self.zoom_out)
        self.menu_bar.add_cascade(label="Edit", menu=self.edit_menu)
        # Image Menu
        self.crop_flag = tk.BooleanVar(value=False)
        self.pick_color_flag = tk.BooleanVar(value=False)
        self.unpick_color_flag = tk.BooleanVar(value=False)
        self.image_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.image_menu.add_command(label="Export Image...", command=self.export_image)
        self.image_menu.add_separator()
        self.image_menu.add_checkbutton(label="Crop", command=self.crop_tool, variable=self.crop_flag)
        self.image_menu.add_checkbutton(label="Pick Color", command=self.pick_color_tool, variable=self.pick_color_flag)
        self.image_menu.add_checkbutton(label="Unpick Color", command=self.unpick_color_tool, variable=self.unpick_color_flag)
        self.image_menu.add_separator()
        self.image_menu.add_command(label="Pick Color Dialog...", command=self.open_pick_color_dialog)
        self.menu_bar.add_cascade(label="Image", menu=self.image_menu)
        # Mak Menu
        self.background_flag = tk.BooleanVar(value=False)
        self.matrix_flag = tk.BooleanVar(value=False)
        self.inclusion_flag = tk.BooleanVar(value=False)
        self.mask_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.mask_menu.add_checkbutton(label="Background", command=self.set_background, variable=self.background_flag)
        self.mask_menu.add_checkbutton(label="Matrix", command=self.set_matrix, variable=self.matrix_flag)
        self.mask_menu.add_checkbutton(label="Inclusion", command=self.set_inclusion, variable=self.inclusion_flag)
        self.menu_bar.add_cascade(label="Mask", menu=self.mask_menu)
        # Attach the menu bar to the root window
        root.config(menu=self.menu_bar)

        # Init entries in menus
        self.file_menu.entryconfig("Save Project", state="disabled")
        self.image_menu.entryconfig("Export Image...", state="disabled")
        self.edit_menu.entryconfig("Zoom in", state="disabled")
        self.edit_menu.entryconfig("Zoom out", state="disabled")
        self.edit_menu.entryconfig("Undo", state="disabled")
        self.edit_menu.entryconfig("Redo", state="disabled")
        self.image_menu.entryconfig("Crop", state="disabled")
        self.image_menu.entryconfig("Pick Color", state="disabled")
        self.image_menu.entryconfig("Unpick Color", state="disabled")
        self.image_menu.entryconfig("Pick Color Dialog...", state="disabled")
        self.mask_menu.entryconfig("Background", state="disabled")
        self.mask_menu.entryconfig("Matrix", state="disabled")
        self.mask_menu.entryconfig("Inclusion", state="disabled")
        
        # Status bar at the bottom of the window
        self.status_frame = tk.Frame(root, bd=1, relief=tk.SUNKEN)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label_size = tk.Label(self.status_frame, text="N/A", anchor=tk.W, width=20)
        self.status_label_size.pack(side=tk.LEFT)
        self.status_label_message = tk.Label(self.status_frame, text="", anchor=tk.W)
        self.status_label_message.pack(fill=tk.X, side=tk.LEFT, expand=True)

        
        
        # Zoom controls
        zoom_frame = tk.Frame(root)
        zoom_frame.pack(pady=5)
        self.image_button = tk.Button(zoom_frame, text="Image", command=self.show_image)
        self.image_button.pack(side=tk.LEFT, padx=2)
        self.image_button.config(state="disabled")
        self.mask_button = tk.Button(zoom_frame, text="Mask", command=self.show_mask)
        self.mask_button.pack(side=tk.LEFT, padx=2)
        self.mask_button.config(state="disabled")
        self.mix_button = tk.Button(zoom_frame, text="Mix", command=self.show_mix)
        self.mix_button.pack(side=tk.LEFT, padx=2)
        self.mix_button.config(state="disabled")
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

        # Info frame
        self.info_frame = tk.Frame(root, width=100, bg="lightgray")
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.info_frame.pack_propagate(False)
        self.x_text_label = tk.Label(self.info_frame, text="X:", font=("Arial", 14), bg="lightgray")
        self.x_text_label.pack(pady=5)
        self.x_label = tk.Label(self.info_frame, text="", font=("Arial", 14), bg="lightgray")
        self.x_label.pack()
        self.y_text_label = tk.Label(self.info_frame, text="Y:", font=("Arial", 14), bg="lightgray")
        self.y_text_label.pack(pady=5)
        self.y_label = tk.Label(self.info_frame, text="", font=("Arial", 14), bg="lightgray")
        self.y_label.pack()
        self.r_text_label = tk.Label(self.info_frame, text="R:", font=("Arial", 14), bg="lightgray")
        self.r_text_label.pack(pady=5)
        self.r_label = tk.Label(self.info_frame, text="", font=("Arial", 14), bg="lightgray")
        self.r_label.pack()
        self.g_text_label = tk.Label(self.info_frame, text="G:", font=("Arial", 14), bg="lightgray")
        self.g_text_label.pack(pady=5)
        self.g_label = tk.Label(self.info_frame, text="", font=("Arial", 14), bg="lightgray")
        self.g_label.pack()
        self.b_text_label = tk.Label(self.info_frame, text="B:", font=("Arial", 14), bg="lightgray")
        self.b_text_label.pack(pady=5)
        self.b_label = tk.Label(self.info_frame, text="", font=("Arial", 14), bg="lightgray")
        self.b_label.pack()

        self.h_text_label = tk.Label(self.info_frame, text="H:", font=("Arial", 14), bg="lightgray")
        self.h_text_label.pack(pady=5)
        self.h_label = tk.Label(self.info_frame, text="", font=("Arial", 14), bg="lightgray")
        self.h_label.pack()

        self.s_text_label = tk.Label(self.info_frame, text="S:", font=("Arial", 14), bg="lightgray")
        self.s_text_label.pack(pady=5)
        self.s_label = tk.Label(self.info_frame, text="", font=("Arial", 14), bg="lightgray")
        self.s_label.pack()

        self.v_text_label = tk.Label(self.info_frame, text="V:", font=("Arial", 14), bg="lightgray")
        self.v_text_label.pack(pady=5)
        self.v_label = tk.Label(self.info_frame, text="", font=("Arial", 14), bg="lightgray")
        self.v_label.pack()


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

        # Init shortcuts
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
        root.bind('<Control-n>', lambda event: self.new_project())
        root.bind('<Control-o>', lambda event: self.load_project())
        root.bind('<Control-q>', lambda event: self.on_closing())
        # Menu edit
        root.bind('<Control-z>', lambda event: self.undo())
        root.bind('<Control-y>', lambda event: self.redo())
        # Tools
        root.bind('<Escape>', lambda event: self.deselect_tool())
        # Display
        root.bind('<Key-1>', lambda event: self.show_image())
        root.bind('<Key-2>', lambda event: self.show_mask())
        root.bind('<Key-3>', lambda event: self.show_mix())
        # Show RGB
        self.canvas.bind("<Motion>", self.show_info)
        
        # Init events
        # Intercept message to close app
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Cancel tool
        self.canvas.bind("<Button-3>", self.deselect_tool)

       
        self.image_id = None

    def set_zoom(self):
        """Set the level of zoom from the value in the box.

        Raises
        ------
        ValueError
            Zoom percentage not positive.
        """
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
        """Reset zoom value to 100%.
        """
        if self.status == STATUS_IMAGE_LOADED:
            self.zoom_level = 1.0
            self.update_zoom_entry()
            self.display_image()

    def zoom_in(self):
        """Routine to increase the zooming level.
        """
        if self.status == STATUS_IMAGE_LOADED:
            self.zoom_level *= 1.2
            self.update_zoom_entry()
            self.display_image()

    def zoom_out(self):
        """Routine to decrease the zooming level.
        """
        if self.status == STATUS_IMAGE_LOADED:
            self.zoom_level /= 1.2
            self.update_zoom_entry()
            self.display_image()

    def update_zoom_entry(self):
        """Routine to update the zoom value in the box.
        """
        if self.status == STATUS_IMAGE_LOADED:
            zoom_percent = int(self.zoom_level * 100)
            self.zoom_entry.delete(0, tk.END)
            self.zoom_entry.insert(0, str(zoom_percent))

        
    def new_project(self):
        """Routine to create a new project."""

        # If the current project is not saved, ask the user if they want to save it.
        if not self.project.saved:
            result = messagebox.askyesnocancel("New Project", "Do you want to save the project before creating a new one?")
            if result is True:
                self.save_project()
                if self.project.saved:
                    self.project = ProjectData()
                else:
                    return
            elif result is False:
                self.project = ProjectData()
        
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.project = ProjectData(image_path = file_path)
            self.change_status(STATUS_IMAGE_LOADED)
            self.zoom_level = 1.0
            self.display_image()
            self.update_image_size_status(self.project.size)
            self.change_status_tool(STATUS_TOOL_NONE)
            self.change_status_display(STATUS_DISPLAY_IMAGE)
            
    
    def update_image_size_status(self, size):
        """Update dimensions of the image in the status bar.

        Parameters
        ----------
        size : tuple
            Integer values describing the dimensions of the image.
        """
        if self.status == STATUS_IMAGE_LOADED:
            self.status_label_size.config(text=f"{size[0]} x {size[1]} pixels")

    def on_click(self, event):
        """Routine to react on left-botton mouse click.
        In case the tool Pick Color is selected, a region is added to the mask.
        In case the tool Unpick Color is selected, a region is removed to the mask.

        Parameters
        ----------
        event : Tkinter event
            Event captured.
        """

        
        if self.status == STATUS_IMAGE_LOADED:
            # Code to pick a color
            if self.status_tool == STATUS_TOOL_PICK_COLOR:
                x, y = int(event.x / self.zoom_level), int(event.y / self.zoom_level)
                width, height = self.project.size
                if 0 <= x < width and 0 <= y < height:
                    # Get the color at the clicked point
                    hue = self.project.hsv[y, x, 0]
                    if hue > 10:
                        lower = np.array([hue-10, 0, 0])
                    else:
                        lower = np.array([0, 0, 0])
                    if hue < 179 - 10:
                        upper = np.array([hue + 10, 255, 255])
                    else:
                        upper = np.array([179, 255, 255])

                    self.project.select_color_hsv(lower, upper, self.current_mask_color)
                    self.update_undo_redo()
                    self.display_image()
            # Code to unpick a color
            elif self.status_tool == STATUS_TOOL_UNPICK_COLOR:
                x, y = int(event.x / self.zoom_level), int(event.y / self.zoom_level)
                width, height = self.project.size
                if 0 <= x < width and 0 <= y < height:
                    # Get the color at the clicked point
                    hue = self.project.hsv[y, x, 0]
                    if hue > 10:
                        lower = np.array([hue-10, 0, 0])
                    else:
                        lower = np.array([0, 0, 0])
                    if hue < 179 - 10:
                        upper = np.array([hue + 10, 255, 255])
                    else:
                        upper = np.array([179, 255, 255])

                    self.project.select_color_hsv(lower, upper, COLOR_MASK_NONE)
                    self.update_undo_redo()
                    self.display_image()


    def show_info(self, event):
        """Routine to update the info on the left panel"""
        if self.status == STATUS_IMAGE_LOADED:
            x, y = int(event.x / self.zoom_level), int(event.y / self.zoom_level)
            width, height = self.project.size
            if 0 <= x < width and 0 <= y < height:
                r, g, b = self.project.rgb.getpixel((x, y))
                h = self.project.hsv[y, x, 0]
                s = self.project.hsv[y, x, 1]
                v = self.project.hsv[y, x, 2]
                self.x_label.config(text=f"{x}")
                self.y_label.config(text=f"{y}")
                self.r_label.config(text=f"{r}")
                self.g_label.config(text=f"{g}")
                self.b_label.config(text=f"{b}")
                self.h_label.config(text=f"{h}")
                self.s_label.config(text=f"{s}")
                self.v_label.config(text=f"{v}")
            
    def display_image(self):
        """Routine to display the image with the correct level of zooming.
        """
        if self.status == STATUS_IMAGE_LOADED:
            if self.display_status == STATUS_DISPLAY_IMAGE:
                width, height = self.project.size
                new_size = (int(width * self.zoom_level), int(height * self.zoom_level))
                resized = self.project.rgb.resize(new_size, Image.LANCZOS) # Resize the image using antialising
                self.tk_image = ImageTk.PhotoImage(resized) # Converts the image into a format Tkinter can use
                
                self.canvas.delete("all") # Clear previous image
                self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
                self.canvas.config(scrollregion=self.canvas.bbox(self.image_id))

            elif self.display_status == STATUS_DISPLAY_MASK:
                width, height = self.project.size
                new_size = (int(width * self.zoom_level), int(height * self.zoom_level))

                mask_rgb = Image.fromarray(self.project.mask)

                resized = mask_rgb.resize(new_size, Image.LANCZOS) # Resize the image using antialising
                self.tk_image = ImageTk.PhotoImage(resized) # Converts the image into a format Tkinter can use
                
                self.canvas.delete("all") # Clear previous image
                self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
                self.canvas.config(scrollregion=self.canvas.bbox(self.image_id))
            elif self.display_status == STATUS_DISPLAY_MIX:
                width, height = self.project.size
                new_size = (int(width * self.zoom_level), int(height * self.zoom_level))

                mask_rgb = Image.fromarray(self.project.mask)
                mix_rgb = Image.blend(self.project.rgb.convert("RGBA"), mask_rgb.convert("RGBA"), 0.5)
                resized = mix_rgb.resize(new_size, Image.LANCZOS) # Resize the image using antialising
                self.tk_image = ImageTk.PhotoImage(resized) # Converts the image into a format Tkinter can use
                
                self.canvas.delete("all") # Clear previous image
                self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
                self.canvas.config(scrollregion=self.canvas.bbox(self.image_id))

    def on_mouse_press(self, event):
        """Routine to start creating the region to be used for cropping the image.

        Parameters
        ----------
        event : Tkinter event
            Event captured.
        """
        if self.status == STATUS_IMAGE_LOADED:
            if self.status_tool == STATUS_TOOL_CROP:
                self.start_x = self.canvas.canvasx(event.x)
                self.start_y = self.canvas.canvasy(event.y)
                if self.rect_id:
                    self.canvas.delete(self.rect_id)
                self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_mouse_drag(self, event):
        """Routine to resize the region to be used for cropping the image.

        Parameters
        ----------
        event : Tkinter event
            Event captured.
        """
        if self.status == STATUS_IMAGE_LOADED:
            if self.status_tool == STATUS_TOOL_CROP:
                cur_x = self.canvas.canvasx(event.x)
                cur_y = self.canvas.canvasy(event.y)
                self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_mouse_release(self, event):
        """Routine to crop the image.

        Parameters
        ----------
        event : Tkinter event
            Event captured.
        """
        if self.status == STATUS_IMAGE_LOADED:
            if self.status_tool == STATUS_TOOL_CROP:
                end_x = self.canvas.canvasx(event.x)
                end_y = self.canvas.canvasy(event.y)

                x1 = int(min(self.start_x, end_x) / self.zoom_level)
                y1 = int(min(self.start_y, end_y) / self.zoom_level)
                x2 = int(max(self.start_x, end_x) / self.zoom_level)
                y2 = int(max(self.start_y, end_y) / self.zoom_level)

                self.project.crop_image(x1, y1, x2, y2)
                self.zoom_level = 1.0
                self.display_image()
                self.update_image_size_status(self.project.size)
                self.file_menu.entryconfig("Save Project", state="normal")
                self.update_undo_redo()
                self.change_status_tool(STATUS_TOOL_NONE)

    def update_undo_redo(self):
        """Routine to update the status of the entries Undo and Redo in the Edit menu.
        """
        if self.status == STATUS_IMAGE_LOADED:
            if self.project.is_undo():
                self.edit_menu.entryconfig("Undo", state="normal")
            else:
                self.edit_menu.entryconfig("Undo", state="disable")
            if self.project.is_redo():
                self.edit_menu.entryconfig("Redo", state="normal")
            else:
                self.edit_menu.entryconfig("Redo", state="disable")

    def undo(self):
        """Undo the last action.
        """
        if self.status == STATUS_IMAGE_LOADED:
            self.project.undo()
            self.display_image()
            self.update_image_size_status(self.project.size)
            if self.project.saved:
                self.file_menu.entryconfig("Save Project", state="disable")
            else: 
                self.file_menu.entryconfig("Save Project", state="normal")
            self.update_undo_redo()
            self.change_status_tool(STATUS_TOOL_NONE)

    def redo(self):
        """Redo the next action.
        """
        if self.status == STATUS_IMAGE_LOADED:
            self.project.redo()
            self.display_image()
            self.update_image_size_status(self.project.size)
            if self.project.saved:
                self.file_menu.entryconfig("Save Project", state="disable")
            else: 
                self.file_menu.entryconfig("Save Project", state="normal")
            self.update_undo_redo()
            self.change_status_tool(STATUS_TOOL_NONE)

    def export_image(self):
        """Routine to save the current image."""
        if self.status == STATUS_IMAGE_LOADED:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                   filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                self.project.rgb.save(file_path)
                self.change_status_tool(STATUS_TOOL_NONE)

    def change_status(self, new_status):
        """Routine to transition between stati for the app.

        Parameters
        ----------
        new_status : integer
            New status
        """

        # If the current and new stati are the same, return becasue no action to do.
        if new_status == self.status:
            return

        if new_status == STATUS_IMAGE_LOADED:
            if self.status == STATUS_NONE:
                self.edit_menu.entryconfig("Zoom in", state="normal")
                self.edit_menu.entryconfig("Zoom out", state="normal")
                self.zoom_entry.config(state="normal")
                self.zoom_in_button.config(state="normal")
                self.zoom_out_button.config(state="normal")
                self.zoom_set_button.config(state="normal")
                self.zoom_reset_button.config(state="normal")
                self.image_menu.entryconfig("Crop", state="normal")
                self.image_menu.entryconfig("Pick Color", state="normal")
                self.image_menu.entryconfig("Unpick Color", state="normal")
                self.image_menu.entryconfig("Export Image...", state="normal")
                self.image_menu.entryconfig("Pick Color Dialog...", state="normal")
                self.mask_menu.entryconfig("Background", state="normal")
                self.mask_menu.entryconfig("Matrix", state="normal")
                self.mask_menu.entryconfig("Inclusion", state="normal")
                self.status = STATUS_IMAGE_LOADED
            else:
                messagebox.showerror("Error", "No transition for the status")

        else:
            messagebox.showerror("Error", "Status unknown")


    def on_closing(self):
        """Event to intercept the closing message and make sure everything is saved."""
        if not self.project.saved:
            result = messagebox.askyesnocancel("Exit", "Do you want to save the project before exiting?")
            self.change_status_tool(STATUS_TOOL_NONE)
            if result is True:
                self.save_project()
                if self.project.saved:
                    self.root.destroy()
            elif result is False:
                self.root.destroy()
            
        else:
            self.root.destroy()

    def deselect_tool(self):
        """Routine to deselect the current selected tool."""
        self.change_status_tool(STATUS_TOOL_NONE)

    def crop_tool(self):
        """Routine to select the crop tool."""
        self.change_status_tool(STATUS_TOOL_CROP)

    def pick_color_tool(self):
        """Routine to select the pick color tool."""
        self.change_status_tool(STATUS_TOOL_PICK_COLOR)
    def unpick_color_tool(self):
        """Routine to select the unpick color tool."""
        self.change_status_tool(STATUS_TOOL_UNPICK_COLOR)

    def change_status_tool(self, new_status):
        """Routine to transition between stati for the app for tools.

        Parameters
        ----------
        new_status : integer
            New status
        """
        # If the current and new stati are the same, return becasue no action to do.
        if new_status == self.status_tool:
            return

        if new_status == STATUS_TOOL_NONE:
            if self.status_tool == STATUS_TOOL_CROP:
                self.start_x = None
                self.start_y = None
                self.rect_id = None
                self.crop_flag.set(False)
                self.canvas.config(cursor="arrow")
                self.status_label_message.config(text="")
                # Events for cropping the image
                self.canvas.unbind("<ButtonPress-1>")
                self.canvas.unbind("<B1-Motion>")
                self.canvas.unbind("<ButtonRelease-1>")
                self.status_tool = STATUS_TOOL_NONE
            elif self.status_tool == STATUS_TOOL_PICK_COLOR:
                self.pick_color_flag.set(False)
                self.canvas.config(cursor="arrow")
                self.status_label_message.config(text="")
                self.canvas.unbind("<Button-1>")
                self.status_tool = STATUS_TOOL_NONE
            elif self.status_tool == STATUS_TOOL_UNPICK_COLOR:
                self.unpick_color_flag.set(False)
                self.canvas.config(cursor="arrow")
                self.status_label_message.config(text="")
                self.canvas.unbind("<Button-1>")
                self.status_tool = STATUS_TOOL_NONE
            else:
                messagebox.showerror("Error", "No transition for the status for tools")
        elif new_status == STATUS_TOOL_CROP:
            # Before chosing a differnt toll, change status to none
            if self.status_tool is not STATUS_TOOL_NONE:
                self.change_status_tool( STATUS_TOOL_NONE)
            if self.status_tool == STATUS_TOOL_NONE:
                self.crop_flag.set(True)
                self.canvas.config(cursor="cross")
                self.status_label_message.config(text="Drag a region to crop the image. Esc to cancel.")
                # Events for cropping the image
                self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
                self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
                self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
                self.status_tool = STATUS_TOOL_CROP
            else:
                messagebox.showerror("Error", "No transition for the status for tools")
        elif new_status == STATUS_TOOL_PICK_COLOR:
            # Before chosing a differnt toll, change status to none
            if self.status_tool is not STATUS_TOOL_NONE:
                self.change_status_tool( STATUS_TOOL_NONE)
            if self.status_tool == STATUS_TOOL_NONE:
                self.pick_color_flag.set(True)
                self.canvas.config(cursor="cross")
                self.status_label_message.config(text="Click on the image to select a color. Esc to cancel.")
                # Event for picking a color
                self.canvas.bind("<Button-1>", self.on_click)
                self.status_tool = STATUS_TOOL_PICK_COLOR
            else:
                messagebox.showerror("Error", "No transition for the status for tools")  
        elif new_status == STATUS_TOOL_UNPICK_COLOR:
            # Before chosing a differnt toll, change status to none
            if self.status_tool is not STATUS_TOOL_NONE:
                self.change_status_tool( STATUS_TOOL_NONE)
            if self.status_tool == STATUS_TOOL_NONE:
                self.unpick_color_flag.set(True)
                self.canvas.config(cursor="cross")
                self.status_label_message.config(text="Click on the image to deselect a color. Esc to cancel.")
                # Event for unpicking a color
                self.canvas.bind("<Button-1>", self.on_click)
                self.status_tool = STATUS_TOOL_UNPICK_COLOR
            else:
                messagebox.showerror("Error", "No transition for the status for tools")   
        else:
            messagebox.showerror("Error", "Unknown status for tools")

    def save_project(self):
        """Routine to save the project."""
        # If the project has never been saved before, ask for the file name.
        if self.project.project_file is None:
            file_path = filedialog.asksaveasfilename(defaultextension=".prj",
                                                filetypes=[("Project files", "*.prj"), ("All files", "*.*")])
            if file_path:
                self.project.saved = True
                self.project.save(file_path)
                self.file_menu.entryconfig("Save Project", state="disabled")
                self.change_status_tool(STATUS_TOOL_NONE)
        else:
            self.project.saved = True
            self.project.save(self.project.project_file)
            self.file_menu.entryconfig("Save Project", state="disabled")
            self.change_status_tool(STATUS_TOOL_NONE)
        self.file_menu.entryconfig("Save Project", state="disabled")

    def save_as_project(self):
        """Routine to save as the project."""
        file_path = filedialog.asksaveasfilename(defaultextension=".prj",
                                            filetypes=[("Project files", "*.prj"), ("All files", "*.*")])
        if file_path:
            self.project.saved = True
            self.project.save(file_path)
            self.file_menu.entryconfig("Save Project", state="disabled")
            self.change_status_tool(STATUS_TOOL_NONE)
        self.file_menu.entryconfig("Save Project", state="disabled")

    def load_project(self):
        """Routine to load a project."""
        # If the current project has been modified, ask the user if they want to save it before opening another project.
        if not self.project.saved:
            result = messagebox.askyesnocancel("Load Project", "Do you want to save the project before loading a new one?")
            if result is True:
                self.save_project()
                if self.project.saved:
                    self.root.destroy()
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Project File", "*.prj")]
        )
        if file_path:
            self.project = ProjectData(project_path = file_path)
            if self.project.rgb is not None:
                self.change_status(STATUS_IMAGE_LOADED)
                self.zoom_level = 1.0
                self.display_image()
                self.update_image_size_status(self.project.size)

        self.file_menu.entryconfig("Save Project", state="disabled")

    def change_status_display(self, new_status):
        """Routine to transition between stati for the display.

        Parameters
        ----------
        new_status : integer
            New status
        """

        if not self.status == STATUS_IMAGE_LOADED:
            return

        # If the current and new stati are the same, return becasue no action to do.
        if new_status == self.display_status:
            return

        if new_status == STATUS_DISPLAY_IMAGE:
            self.display_status = STATUS_DISPLAY_IMAGE
            self.image_button.config(state="disabled")
            self.mask_button.config(state="normal")
            self.mix_button.config(state="normal")
        elif new_status == STATUS_DISPLAY_MASK:
            self.display_status = STATUS_DISPLAY_MASK
            self.image_button.config(state="normal")
            self.mask_button.config(state="disabled")
            self.mix_button.config(state="normal")
        elif new_status == STATUS_DISPLAY_MIX:
            self.display_status = STATUS_DISPLAY_MIX
            self.image_button.config(state="normal")
            self.mask_button.config(state="normal")
            self.mix_button.config(state="disabled")
            
        else:
            messagebox.showerror("Error", "Unknown display status for tools")

        self.display_image()

    def show_image(self):
        """Routine to show the image."""
        self.change_status_display(STATUS_DISPLAY_IMAGE)

    def show_mask(self):
        """Routine to show the mask."""
        self.change_status_display(STATUS_DISPLAY_MASK)

    def show_mix(self):
        """Routine to show the blend between the image and the mask."""
        self.change_status_display(STATUS_DISPLAY_MIX)
        
    def set_background(self):
        """Routine to set to mark the background on the mask.
        """
        self.background_flag.set(True)
        self.matrix_flag.set(False)
        self.inclusion_flag.set(False)
        self.current_mask_color = COLOR_MASK_BACKGROUND
        
    def set_matrix(self):
        """Routine to set to mark the matrix on the mask.
        """
        self.background_flag.set(False)
        self.matrix_flag.set(True)
        self.inclusion_flag.set(False)
        self.current_mask_color = COLOR_MASK_MATRIX
        
    def set_inclusion(self):
        """Routine to set to mark the inclusion on the mask.
        """
        self.background_flag.set(False)
        self.matrix_flag.set(False)
        self.inclusion_flag.set(True)
        self.current_mask_color = COLOR_MASK_INCLUSION
        
    def open_pick_color_dialog(self):
        """Routine to open the dialogue to set the parameters for the tool pick color."""
        PickColorDialog(self.root, self.pick_color_dialog_receive_values, self.pick_color_params)
        
    def pick_color_dialog_receive_values(self, values, mode):
        """Routine to process the values from the dialogue to set the parameters for the tool pick color."""
        if values is not None:
            if mode == 'HSV':
                self.pick_color_params['mode'] = 'HSV'
                self.pick_color_params['H'] = values['H']
                self.pick_color_params['S'] = values['S']
                self.pick_color_params['V'] = values['V']
            else:
                self.pick_color_params['mode'] = 'RGB'
                self.pick_color_params['R'] = values['R']
                self.pick_color_params['G'] = values['G']
                self.pick_color_params['B'] = values['B']


# Run the app
root = tk.Tk()
app = ImageViewer(root)
root.mainloop()
