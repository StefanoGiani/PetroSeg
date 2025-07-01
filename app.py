# Python app to crop an image.
#
# Features:
# - The image can be zoomed in and out.
# - Crop an image dragging a region with the mouse.
# - Undo and redo.
# - Added mechanism for tools.
# - Added message label in the bottom status bar.

# Shortcuts:
# Ctrl+N New project and open an image.
# Ctrl+S Save project.
# Ctrl+O Load project.
# Ctrl+Q Terminate the app.
# +, - Zoom level.
# Arrow keys Pan the image.
# Ctrl+Z Undo.
# Ctrl+Y Redo.
# Esc Deselect current tool

# TODO:
# - Masking background.

# Author Stefano Giani

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pickle

# Stati of the app:
STATUS_NONE = 0 # Just started 
STATUS_IMAGE_LOADED = 1 # An image is loaded

# Stati of the tools
STATUS_TOOL_NONE = 0 # No tool selected.
STATUS_TOOL_CROP = 1 # Crop selected

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
        
            self.rgb = Image.open(image_path)
            self.size = self.rgb.size
            self.saved = False
            self.stack_actions = []
            self.stack_actions.append({"action": "LoadImage(" + image_path + ")", "saved": self.saved, "size": self.size, "rgb": self.rgb.copy()})
            self.cursor_stack_action = 0
            self.project_file = None
        elif project_path is not None:
            self.load(project_path)
        else:
            self.rgb = None
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
            state['size'] = self.size
            state['stack_actions'] = self.stack_actions
            state['cursor_stack_action'] = self.cursor_stack_action
            state['project_file'] = project_path
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
            self.size = state['size'] 
            self.stack_actions = state['stack_actions']
            self.cursor_stack_action = state['cursor_stack_action']
            self.project_file = state['project_file']
        self.saved = True



        with open(project_path, "wb") as f:
            state= {}
            state['rgb'] = self.rgb
            state['size'] = self.size
            state['stack_actions'] = self.stack_actions
            state['cursor_stack_action'] = self.cursor_stack_action
            pickle.dump(state, f)
        self.saved = True


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
        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        self.rgb = self.rgb.crop((x1, y1, x2, y2))
        self.size = self.rgb.size
        self.saved = False
        self.stack_actions.append({"action": "CropImage(" + str(x1) + "," + str(y1) + "," +  str(x2) + "," +  str(y2) + ")", "saved": self.saved, "size": self.size, "rgb": self.rgb.copy()})

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
        self.image_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.image_menu.add_command(label="Export Image...", command=self.export_image)
        self.image_menu.add_separator()
        self.image_menu.add_checkbutton(label="Crop", command=self.crop_tool, variable=self.crop_flag)
        self.menu_bar.add_cascade(label="Image", menu=self.image_menu)
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

        # Init events
        # Events for cropping the image
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        # Intercept message to close app
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
       
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
            
    
    def update_image_size_status(self, size):
        """Update dimensions of the image in the status bar.

        Parameters
        ----------
        size : tuple
            Integer values describing the dimensions of the image.
        """
        if self.status == STATUS_IMAGE_LOADED:
            self.status_label_size.config(text=f"{size[0]} x {size[1]} pixels")
            
    def display_image(self):
        """Routine to display the image with the correct level of zooming.
        """
        if self.status == STATUS_IMAGE_LOADED:
            width, height = self.project.size
            new_size = (int(width * self.zoom_level), int(height * self.zoom_level))
            resized = self.project.rgb.resize(new_size, Image.LANCZOS) # Resize the image using antialising
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
                self.image_menu.entryconfig("Export Image...", state="normal")
                self.status = STATUS_IMAGE_LOADED
            else:
                messagebox.showerror("Error", "No transition for the status")

        else:
            messagebox.showerror("Error", "Status unknown")


    def on_closing(self):
        """Event to intercept the closing message and make sure everything is saved."""
        if not self.project.saved:
            result = messagebox.askyesnocancel("Exit", "Do you want to save the project before exiting?")
            if result is True:
                self.save_project()
                if self.project.saved:
                    self.root.destroy()
            elif result is False:
                self.root.destroy()
            self.change_status_tool(STATUS_TOOL_NONE)
        else:
            self.root.destroy()

    def deselect_tool(self):
        """Routine to deselect the current selected tool."""
        self.change_status_tool(STATUS_TOOL_NONE)

    def crop_tool(self):
        """Routine to select the crop tool."""
        self.change_status_tool(STATUS_TOOL_CROP)

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
                self.status_tool = STATUS_TOOL_NONE
            else:
                messagebox.showerror("Error", "No transition for the status for tools")
        elif new_status == STATUS_TOOL_CROP:
            if self.status_tool == STATUS_TOOL_NONE:
                self.crop_flag.set(True)
                self.canvas.config(cursor="cross")
                self.status_label_message.config(text="Drag a region to crop the image. Esc to cancel.")
                self.status_tool = STATUS_TOOL_CROP
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



# Run the app
root = tk.Tk()
app = ImageViewer(root)
root.mainloop()
