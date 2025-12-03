
# This file is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# See the LICENSE file or visit https://creativecommons.org/licenses/by-nc/4.0/ for details.
# © Stefano Giani


# Python app to label images of thin sections for segmentation.
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
# - Ruler to measure features on the image and to set conversion between pixels and real units.
# - Automatic method to find regions based on edges and other criteria.
# - Select a connected region click with the mouse based on color similarity.
# - Select using color range.
# - Show histograms of channels.
# - Tool to deselect a region on the mask.
# - Tool to select a rectangular region on the mask.
# - Tool to deselect a rectangular region on the mask.
# - Possibility to decide if to overwride already set classes in the mask or not.

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
# 1. Cerate filter to filter out already creted regions accoring to characteristics.
# 2. Allow to export mask as an image with/without boundary boxes for inclusions.

# Authors Lucy Standish and Stefano Giani

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pickle
import cv2
import numpy as np
import math 
import matplotlib.pyplot as plt
import colorsys
import h5py
import os
from pathlib import Path

# Debugging flags:
CONSOLE_DEBUG = True # Flag to activate debugging information in consol

# Stati of the app:
STATUS_NONE = 0 # Just started 
STATUS_IMAGE_LOADED = 1 # An image is loaded

# Stati of the tools
STATUS_TOOL_NONE = 0 # No tool selected.
STATUS_TOOL_CROP = 1 # Crop selected
STATUS_TOOL_PICK_COLOR = 2 # Pick color selected
STATUS_TOOL_UNPICK_COLOR = 3 # Unpick color selected
STATUS_TOOL_RULER = 4 # Ruler selected
STATUS_TOOL_PICK_REGION = 5 # Pick region selected
STATUS_TOOL_UNPICK_REGION = 6 # Unpick region selected
STATUS_TOOL_PICK_RECT = 7 # Pick a rectangular region
STATUS_TOOL_UNPICK_RECT = 8 # Unpick a rectangular region

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
    def __init__(self, project_path = None, image_path = None, clear_stack_flag = False, console_debug = False ):
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
        clear_stack_flag : logical
            Flag to clear the stack of actions for the loaded project, by default False.
        console_debug : logical
            Flag to output debugging information to the console, by default False.
        """
        self.console_debug = console_debug
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
            self.unit = "px"
            self.conversion_factors = {"px": 1.0, "cm": None, "mm": None, "in": None}
        elif project_path is not None:
            self.saved = True
            self.load(project_path, clear_stack_flag)
        else:
            self.rgb = None
            self.hsv = None
            self.mask = None
            self.size = None
            self.saved = True
            self.stack_actions = []
            self.cursor_stack_action = -1
            self.project_file = None
            self.unit = "px"
            self.conversion_factors = {"px": 1.0, "cm": None, "mm": None, "in": None}
        if self.console_debug:
            self.console_debug_stack_actions()

    def console_debug_stack_actions(self):
        print('cursor_stack_action:',self.cursor_stack_action)
        for i in range(len(self.stack_actions)):
            if i == self.cursor_stack_action:
                print('---> action: ', self.stack_actions[i]["action"])
            else:
                print('     action: ', self.stack_actions[i]["action"])
        print('----------------------------------------------')
            
    def snapshot(self):
        """Take a snapshot of the current object."""
        self.truncate_stack()
        self.stack_actions.append({"action": "Snapshot()", "saved": self.saved, "size": self.size, "rgb": self.rgb.copy(), "hsv": self.hsv.copy(), 
                                       "mask":self.mask.copy()})
        self.cursor_stack_action += 1
        if self.console_debug:
                self.console_debug_stack_actions()

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
            state['unit'] = self.unit
            state['conversion_factors'] = self.conversion_factors
            self.project_file = project_path
            pickle.dump(state, f)
        self.saved = True

    def load(self, project_path, clear_stack_flag = False):
        """Routine to load a project from disk.

        Parameters
        ----------
        project_path : string
            Path to the file containing the project to open.
        clear_stack_flag : logical
            Flag to clear the stack of actions for the loaded project, by default False.
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
            self.unit = state['unit']
            self.conversion_factors = state['conversion_factors']
        if clear_stack_flag:
            self.clear_stack()
            self.snapshot()
        self.saved = True
        if self.console_debug:
                self.console_debug_stack_actions()


    def select_color_hsv(self, lower, upper, color, overwrite = True):
        """Select a region in the mask based on range in HSV colors.
        
        Parameters
        ----------
        lower : tuple
            Three uint8 values to determine the lower value for the range in HSV.
        upper : tuple
            Three uint8 values to determine the upper value for the range in HSV.
        color : tuple
            Three uint8 values to determine the RGB color to use to mark the region on the mask.
        overwrite : logical, optional
            Control if the non zero values of the mask can be overwritten, by default True."""
        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        mask = cv2.inRange(self.hsv, lower, upper)
        if overwrite:
            # Apply the color where mask == 255
            self.mask[mask == 255] = color
        else:
            # Create a boolean mask for the condition
            condition = (self.mask[:,:,0] == 0) & (self.mask[:,:,1] == 0) & (self.mask[:,:,2] == 0) & (mask == 255)
            self.mask[condition] = color
        self.saved = False
        self.truncate_stack()
        self.stack_actions.append({"action": "SelectColorHSV(" + str(lower) + "," + str(upper)  + "," + str(color)  + "," + str(overwrite) + ")", "saved": self.saved, "mask": self.mask.copy()})
        if self.console_debug:
                self.console_debug_stack_actions()
        
    def select_color_rgb(self, lower, upper, color, overwrite = True):
        """Select a region in the mask based on range in RGB colors.
        
        Parameters
        ----------
        lower : tuple
            Three uint8 values to determine the lower value for the range in RGB.
        upper : tuple
            Three uint8 values to determine the upper value for the range in RGB.
        color: tuple
            Three uint8 values to determine the RGB color to use to mark the region on the mask.
        overwrite : logical, optional
            Control if the non zero values of the mask can be overwritten, by default True."""
        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        mask = cv2.inRange(np.array(self.rgb), lower, upper)
        if overwrite:
            # Apply the color where mask == 255
            self.mask[mask == 255] = color
        else:
            # Create a boolean mask for the condition
            condition = (self.mask[:,:,0] == 0) & (self.mask[:,:,1] == 0) & (self.mask[:,:,2] == 0) & (mask == 255)
            self.mask[condition] = color
        self.saved = False
        self.truncate_stack()
        self.stack_actions.append({"action": "SelectColorRGB(" + str(lower) + "," + str(upper)  + "," + str(color)  + "," + str(overwrite) + ")", "saved": self.saved, "mask": self.mask.copy()})
        if self.console_debug:
                self.console_debug_stack_actions()
        
    def select_region_hsv(self, point, lower, upper, color, overwrite = True):
        """Select a connected region in the mask using HSV.
        The floofill routine in cv2 accepts only RGB images, not HSV.
        Therefore I have to do a workaround by selecting a color range in HSV first,
        and then find the connected region with the selected point.
        
        Parameters
        ----------
        point : tuple
            Coordinate of the region.
        lower : tuple
            Three uint8 values to determine the lower value for the range in HSV.
        upper : tuple
            Three uint8 values to determine the upper value for the range in HSV.
        color: tuple
            Three uint8 values to determine the RGB color to use to mark the region on the mask.
        overwrite : logical, optional
            Control if the non zero values of the mask can be overwritten, by default True."""
        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        color_tmp = self.hsv[point[1],point[0],:]
        color_lower = color_tmp - lower
        color_upper = color_tmp + upper
        if color_lower[0]<0:
            color_lower[0] = 0
        if color_lower[1]<0:
            color_lower[1] = 0
        if color_lower[2]<0:
            color_lower[2] = 0
        if color_upper[0]>179:
            color_upper[0] = 179
        if color_upper[1]>255:
            color_upper[1] = 255
        if color_upper[2]>255:
            color_upper[2] = 255
        mask = cv2.inRange(self.hsv, color_lower, color_upper)
        
        # Create a mask for floodFill (must be 2 pixels larger)
        flood_mask = np.zeros((self.size[1]+2, self.size[0]+2), np.uint8)
        
        
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Flood fill with a temporary color
        cv2.floodFill(mask_rgb, flood_mask, point, (255,255,255),
                      loDiff=(0,0,0), upDiff=(0,0,0), flags=4)
        
        filled_region = flood_mask[1:-1, 1:-1]
        
        if overwrite:
            # Create a boolean mask for the condition
            condition = filled_region != 0
        else:
            # Create a boolean mask for the condition
            condition = (self.mask[:,:,0] == 0) & (self.mask[:,:,1] == 0) & (self.mask[:,:,2] == 0) & (filled_region != 0)

        self.mask[condition] = color
        self.saved = False
        self.truncate_stack()
        self.stack_actions.append({"action": "SelectRegionHSV(" + str(point) + "," + str(lower) + "," + str(upper)  + "," + str(color)  + "," + str(overwrite)  + ")", "saved": self.saved, "mask": self.mask.copy()})
        if self.console_debug:
                self.console_debug_stack_actions()
        
    def select_region_rgb(self, point, lower, upper, color, overwrite = True):
        """Select a connected region in the mask using RGB.
        
        Parameters
        ----------
        point : tuple
            Coordinate of the region.
        lower : tuple
            Three uint8 values to determine the lower value for the range in RGB.
        upper : tuple
            Three uint8 values to determine the upper value for the range in RGB.
        color: tuple
            Three uint8 values to determine the RGB color to use to mark the region on the mask.
        overwrite : logical, optional
            Control if the non zero values of the mask can be overwritten, by default True."""
        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        mask_floodfill = np.zeros((self.size[1]+2, self.size[0]+2), np.uint8)
        cv2.floodFill(np.array(self.rgb), mask_floodfill, point, (255,255,255),
                      loDiff=lower, upDiff=upper, flags=4)
        filled_region = mask_floodfill[1:-1, 1:-1]
        
        if overwrite:
            # Create a boolean mask for the condition
            condition = filled_region != 0
        else:
            # Create a boolean mask for the condition
            condition = (self.mask[:,:,0] == 0) & (self.mask[:,:,1] == 0) & (self.mask[:,:,2] == 0) & (filled_region != 0)

        self.mask[condition] = color
        self.saved = False
        self.truncate_stack()
        self.stack_actions.append({"action": "SelectRegionRGB(" + str(point) + "," + str(lower) + "," + str(upper)  + "," + str(color)  + "," + str(overwrite) + ")", "saved": self.saved, "mask": self.mask.copy()})
        if self.console_debug:
                self.console_debug_stack_actions()
        
    def deselect_region(self, point):
        """Deselect a connected region in the mask.
        
        Parameters
        ----------
        point : tuple
            Coordinate of the region."""
        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        mask_floodfill = np.zeros((self.size[1]+2, self.size[0]+2), np.uint8)
        cv2.floodFill(self.mask, mask_floodfill, point, (255,255,255),
                      loDiff=(0,0,0), upDiff=(0,0,0), flags=4)
        filled_region = mask_floodfill[1:-1, 1:-1]
        
        # Create a boolean mask for the condition
        condition = filled_region != 0

        self.mask[condition] = (0,0,0)
        self.saved = False
        self.truncate_stack()
        self.stack_actions.append({"action": "DeselectRegion(" + str(point) + ")", "saved": self.saved, "mask": self.mask.copy()})
        if self.console_debug:
                self.console_debug_stack_actions()
        

    def truncate_stack(self):
        """Truncate the stack of actions.

        This can be used when an action is performed on the data and the remaining redo actions are no longer applicable.
        """
        if self.cursor_stack_action >= 0:
            del self.stack_actions[self.cursor_stack_action+1:]
        if self.console_debug:
                self.console_debug_stack_actions()
            
    def clear_stack(self):
        """Claer the stack of actions.

        """
        self.cursor_stack_action = -1
        self.stack_actions = []
        if self.console_debug:
                self.console_debug_stack_actions()

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
        self.truncate_stack()
        self.stack_actions.append({"action": "CropImage(" + str(x1) + "," + str(y1) + "," +  str(x2) + "," +  str(y2) + ")", "saved": self.saved, 
                                   "size": self.size, "rgb": self.rgb.copy(), "hsv": self.hsv.copy(), "mask": self.mask.copy()})
        if self.console_debug:
                self.console_debug_stack_actions()
        

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

        if self.console_debug:
                self.console_debug_stack_actions()
            
        
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

        if self.console_debug:
                self.console_debug_stack_actions()

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
        
    def find_regions(self, mask_color, range_color, color_encoding='RGB', edge_threshold1=100,
                        edge_threshold2=200, min_area=1, max_area=1000,
                        ratio_threshold=0.8, contour_retrieval=cv2.RETR_TREE,
                        approximation=cv2.CHAIN_APPROX_SIMPLE, preview=None, overwrite = False):
        """Routine to automatically find regions using edges.

        Parameters
        ----------
        mask_color : tuple
            Three uint8 values to determine the RGB color to use to mark the region on the mask.
        range_color : list of two tuple
            Two sets of three uint8 values each to determine the lower and upper values for the range in RGB/HSV.
        color_encoding : str, optional
            Type of color encoding used in range_color. Either RGB or HSV, by default 'RGB'
        edge_threshold1 : int, optional
            Lower bound for the gradient intensity for detecting edges, by default 100
        edge_threshold2 : int, optional
            Upper bound for the gradient intensity for detecting edges, by default 200
            Edges between threshold1 and threshold2 are accepted only if they are connected to strong edges (above edge_threshold2).
        min_area : int, optional
            Minimum value for the area in pixels for an iclusion to be considered, by default 1
        max_area : int, optional
            Maximum value for the area in pixels for an iclusion to be considered, by default 1000
        ratio_threshold : float, optional
            Threshold for the ratio between area and hull area of regions. regions above the threshold are cosidered, by default 0.8
        contour_retrieval : cv2 constant, optional
            Specifies the hierarchy of contours to retrieve, by default cv2.RETR_TREE
            cv2.RETR_EXTERNAL: Retrieves only the outermost contours. Used when interested in the external shape of objects.
            cv2.RETR_LIST: Retrieves all contours without establishing any hierarchical relationships. When you want all contours regardless of nesting, and hierarchy is not important.
            cv2.RETR_TREE: Retrieves all contours and reconstructs a full hierarchy of nested contours. When you need to understand the structure of nested shapes (e.g., holes inside objects).
            cv2.RETR_CCOMP: Retrieves all contours and organizes them into a two-level hierarchy (outer and inner). When you want to distinguish between objects and their holes, but don’t need full nesting.
        approximation : cv2 constantpe, optional
            Specifies how the contour points are stored:, by default cv2.CHAIN_APPROX_SIMPLE
            cv2.CHAIN_APPROX_NONE: Stores all the boundary points. This can be memory-intensive.
            cv2.CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments and keeps only their end points. This is more efficient and commonly used.
        preview : string, optional
            Specify if the user wants just a preview of one of the itnermediate steps or run the algorithm, by default None:
            None: Full algorithm
            "edges" : mask with edges returned.
            "colors" : return mask of selected colors.
        overwrite : logical, optional
            Control if the non zero values of the mask can be overwritten, by default False.

        Returns
        -------
        integer
            Number of regions found if the full algorithm is run.
        mask
            if a preview is selected.
        """
        
        if preview is None:
            self.truncate_stack()
            self.cursor_stack_action = len(self.stack_actions)
        
        # Convert image to greyscale
        image = np.array(self.rgb)
        if (preview == "edges") or (preview is None):
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Apply edge detection
            # Fine-tuning the thresholds in the cv2.Canny() function can help. 
            # Lower thresholds might detect more edges, while higher thresholds might reduce noise.
            edges = cv2.Canny(gray_image, edge_threshold1, edge_threshold2)
            #cv2.imshow('image',image)
            #cv2.imshow('edges',edges)
            #cv2.waitKey(0)
        if preview == "edges":
            return edges
        
        if (preview == "colors") or (preview is None):
            # Create mask
            if color_encoding=='RGB':
                mask = cv2.inRange(image, np.asarray(range_color[0]), np.asarray(range_color[1])) 
            else: # HSV
                mask = cv2.inRange(self.hsv, np.asarray(range_color[0]), np.asarray(range_color[1])) 
        if preview == "colors":
            return mask

        # Find contours
        # Experiment with different methods for contour retrieval (cv2.RETR_EXTERNAL, cv2.RETR_TREE, etc.) 
        # and approximation (cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE).
        contours, _ = cv2.findContours(edges, contour_retrieval, approximation)

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
            inclusions = cv2.findNonZero(inclusion_mask)
            # add masks
            mask_incusions += inclusion_mask
            if inclusions is not None:
                inclusion_count += 1
        # Clip mask values        
        mask_incusions = mask_incusions.clip(0, 255).astype("uint8")
                
        if overwrite:
            # Apply the color where mask == 255
            self.mask[mask_incusions != 0] = mask_color
        else:
            # Create a boolean mask for the condition
            condition = (self.mask[:,:,0] == 0) & (self.mask[:,:,1] == 0) & (self.mask[:,:,2] == 0) & (mask_incusions != 0)
            self.mask[condition] = mask_color
        self.saved = False
        self.truncate_stack()
        self.stack_actions.append({"action": "FindRegions(" + str(mask_color)  + 
                                   "," + str(range_color)  + "," + str(color_encoding)  + 
                                   "," + str(edge_threshold1)  + "," + str(edge_threshold2)  + 
                                   "," + str(min_area)  + "," + str(max_area)  + 
                                   "," + str(ratio_threshold)  + "," + str(contour_retrieval)  + 
                                   "," + str(approximation) + "," + str(overwrite) + ")", "saved": self.saved, "mask": self.mask.copy()})
        if self.console_debug:
                self.console_debug_stack_actions()
                                   
        return inclusion_count   
    
    def color_range(self, mask_color, range_color, color_encoding='RGB', overwrite = True):
        """Routine to select a range of colors.

        Parameters
        ----------
        mask_color : tuple
            Three uint8 values to determine the RGB color to use to mark the region on the mask.
        range_color : list of two tuple
            Two sets of three uint8 values each to determine the lower and upper values for the range in RGB/HSV.
        color_encoding : str, optional
            Type of color encoding used in range_color. Either RGB or HSV, by default 'RGB'.
        overwrite : logical, optional
            Control if the non zero values of the mask can be overwritten, by default True.
        """
        
        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        
        if color_encoding=='RGB':
            image = np.array(self.rgb)
            mask = cv2.inRange(image, np.asarray(range_color[0]), np.asarray(range_color[1])) 
        else: # HSV
            mask = cv2.inRange(self.hsv, np.asarray(range_color[0]), np.asarray(range_color[1])) 
        if overwrite:
            # Apply the color where mask == 255
            self.mask[mask != 0] = mask_color
        else:
            # Create a boolean mask for the condition
            condition = (self.mask[:,:,0] == 0) & (self.mask[:,:,1] == 0) & (self.mask[:,:,2] == 0) & (mask != 0)
            self.mask[condition] = mask_color
            
        self.saved = False
        self.truncate_stack()
        self.stack_actions.append({"action": "ColorRange(" + str(mask_color)  + 
                                   "," + str(range_color)  + "," + str(color_encoding) + "," + str(overwrite) + ")", "saved": self.saved, "mask": self.mask.copy()})
        if self.console_debug:
                self.console_debug_stack_actions()
        
    def mask_rectangle(self, start_point, end_point, color, overwrite = True):
        """Select a rectangular region on the mask.
        
        Parameters
        ----------
        start_point : tuple
            Coordinate of the first corner of the rectangular region.
        end_point : tuple
            Coordinate of the second corner of the rectangular region.
        color: tuple
            Three uint8 values to determine the RGB color to use to mark the region on the mask.
        overwrite : logical, optional
            Control if the non zero values of the mask can be overwritten, by default True."""
        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        
        if overwrite:
            cv2.rectangle(self.mask, start_point, end_point, color, thickness=-1)
        else:
            mask_copy = self.mask.copy()
            cv2.rectangle(mask_copy, start_point, end_point, color, thickness=-1)
            condition = (mask_copy[:,:,0] == color[0]) & (mask_copy[:,:,1] == color[1]) & (mask_copy[:,:,2] == color[2]) & (self.mask[:,:,0] == 0) & (self.mask[:,:,1] == 0) & (self.mask[:,:,2] == 0)
            self.mask[condition] = color
        self.saved = False
        self.truncate_stack()
        self.stack_actions.append({"action": "MaskRectangle(" + str(start_point) + "," + str(end_point) + "," + str(color) + "," + str(overwrite) + ")", "saved": self.saved, "mask": self.mask.copy()})
        if self.console_debug:
                self.console_debug_stack_actions()
                  
    def unmask_rectangle(self, start_point, end_point):
        """Deselect a rectangular region on the mask.
        
        Parameters
        ----------
        start_point : tuple
            Coordinate of the first corner of the rectangular region.
        end_point : tuple
            Coordinate of the second corner of the rectangular region."""
        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        
        cv2.rectangle(self.mask, start_point, end_point, (0,0,0), thickness=-1)
        self.saved = False
        self.truncate_stack()
        self.stack_actions.append({"action": "UnmaskRectangle(" + str(start_point) + "," + str(end_point) + ")", "saved": self.saved, "mask": self.mask.copy()})
        if self.console_debug:
                self.console_debug_stack_actions()
                  
    def check_mask_consistency(self):
        """Routine to check if in the mask there are still pixels not associated to a class.
        If all pixels associated to a class, then the mask is consistent and the routine returns true.

        Returns
        -------
        logical
            If all pixels associated to a class, then the mask is consistent and the routine returns true.
        """
        target_color = np.array([0, 0, 0]) 

        mask = cv2.inRange(self.mask, target_color, target_color)
        
        return (not (cv2.countNonZero(mask) > 0))
    
    def select_color_mask(self, lower, upper, color, overwrite = True):
        """Select a region in the mask based on range in RGB colors of the mask.
        
        Parameters
        ----------
        lower : tuple
            Three uint8 values to determine the lower value for the range in RGB.
        upper : tuple
            Three uint8 values to determine the upper value for the range in RGB.
        color: tuple
            Three uint8 values to determine the RGB color to use to mark the region on the mask.
        overwrite : logical, optional
            Control if the non zero values of the mask can be overwritten, by default True."""
        self.truncate_stack()
        self.cursor_stack_action = len(self.stack_actions)
        mask = cv2.inRange(self.mask, lower, upper)
        if overwrite:
            # Apply the color where mask == 255
            self.mask[mask == 255] = color
        else:
            # Create a boolean mask for the condition
            condition = (self.mask[:,:,0] == 0) & (self.mask[:,:,1] == 0) & (self.mask[:,:,2] == 0) & (mask == 255)
            self.mask[condition] = color
        self.saved = False
        self.truncate_stack()
        self.stack_actions.append({"action": "SelectColorMask(" + str(lower) + "," + str(upper)  + "," + str(color)  + "," + str(overwrite) + ")", "saved": self.saved, "mask": self.mask.copy()})
        if self.console_debug:
                self.console_debug_stack_actions()
        
    def extract_regions(self, color):
        """Routine to compute statistics of the inclusions.

        Parameters
        ----------
        color: tuple
            Three uint8 values to determine the RGB color used in the mask to indicate inclusions.

        Returns
        -------
        list
            Each element of the list represent an inclusion and for each inclusions the following informations are reported:
            area
            boundary box
            centroid
            aspect_ratio
            circularity
            convex_hull_area
            solidity
            eccentricity
            convexity
            hu_moments
            avg_rgb
            avg_hsv
            std_rgb
            std_hsv
            
        """

        # Create a mask for the inclusion color
        mask = cv2.inRange(self.mask, np.array(color), np.array(color))


        # Find contours of the disconnected regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract statistics for each region
        regions_stats = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area != 0 else 0
            convexity = perimeter / cv2.arcLength(hull, True) if cv2.arcLength(hull, True) != 0 else 0
    
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0
                
                
            # Eccentricity calculation using ellipse fitting
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0
                
            # Hu Moments
            hu_moments = cv2.HuMoments(M).flatten().tolist()
            
            #for contour in contours:
            # Create a blank mask the same size as the original mask
            region_mask = np.zeros(mask.shape, dtype=np.uint8)

            # Draw the filled contour on the mask
            cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)

            # Get coordinates of all non-zero pixels (i.e., the region)
            pixels = np.column_stack(np.where(region_mask == 255))

            # Convert to list of (x, y) tuples
            pixel_list = [(int(x), int(y)) for y, x in pixels]
            
            # Extract RGB values for each pixel
            rgb_values = [self.rgb.getpixel((x, y)) for x, y in pixel_list]

            # Compute average and std RGB
            avg_rgb = tuple(np.mean(rgb_values, axis=0))
            std_rgb = np.std(rgb_values, axis=0)

            # Convert RGB values to HSV (normalized to [0, 1])
            hsv_values = [colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0) for r, g, b in rgb_values]

            # Compute average and std HSV
            avg_hsv = tuple(np.mean(hsv_values, axis=0))
            std_hsv = np.std(hsv_values, axis=0)

            # Convert average HSV back to 0–255 scale for S and V
            avg_hsv_scaled = (avg_hsv[0], avg_hsv[1] * 255, avg_hsv[2] * 255)
            std_hsv_scaled = (std_hsv[0], std_hsv[1] * 255, std_hsv[2] * 255)

            # Discard regions with area 0
            if area > 0:
                regions_stats.append({
                    'area': area,
                    'bounding_box': (x, y, w, h),
                    'centroid': (cx, cy),
                    'aspect_ratio': aspect_ratio,
                    'circularity': circularity,
                    'convex_hull_area': hull_area,
                    'solidity': solidity,
                    'eccentricity': eccentricity,
                    'convexity': convexity,
                    'hu_moments': hu_moments,
                    'avg_rgb': avg_rgb,
                    'avg_hsv': avg_hsv_scaled,
                    'std_rgb': std_rgb,
                    'std_hsv': std_hsv_scaled
                })
            
        return regions_stats
    
    def save_regions_stats_pickle(self, regions_stats, file_name):
        """Routine to save the statistics of the regions using pickle.

        Parameters
        ----------
        regions_stats : list
           Each element of the list represent an inclusion and for each inclusions the following informations are reported:
            area
            boundary box
            centroid
            aspect_ratio
            circularity
            convex_hull_area
            solidity
            eccentricity
            convexity
            hu_moments
            avg_rgb
            avg_hsv
            std_rgb
            std_hsv
        file_name : string
            Name of the file to use.
        """
        
        with open(file_name, "wb") as f:
            pickle.dump(regions_stats, f)

    def save_regions_stats_h5(self, regions_stats, file_name):
        """Routine to save the statistics of the regions using h5 format.

        Parameters
        ----------
        regions_stats : list
           Each element of the list represent an inclusion and for each inclusions the following informations are reported:
            area
            boundary box
            centroid
            aspect_ratio
            circularity
            convex_hull_area
            solidity
            eccentricity
            convexity
            hu_moments
            avg_rgb
            avg_hsv
            std_rgb
            std_hsv
        file_name : string
            Name of the file to use.
        """
        
        with h5py.File(file_name, "w") as h5f:
            for i, region in enumerate(regions_stats):
                grp = h5f.create_group(f"region_{i+1}")
                for key, value in region.items():
                    if isinstance(value, (list, tuple)):
                        grp.create_dataset(key, data=np.array(value))
                    else:
                        grp.attrs[key] = value
                        
    def create_tiles(self, tile_width, tile_height, overlap_x, overlap_y, tiles_name, output_dir):
        """Routine to create tiles from the pair image and mask.
        Tiles are named as {tiles_name}_{tile_count}.png.
        Masks for the tiles are named {tiles_name}_mask_{tile_count}.png.

        Parameters
        ----------
        tile_width : integer
            Horizontal dimension to the tiles.
        tile_height : integer
            Vertical dimension to the tiles.
        overlap_x : integer
            Number of pixels of the horizontal overlay of tiles
        overlap_y : integer
            Number of pixels of the vertical overlay of tiles
        tiles_name : string
            Root name for the tiles.
        output_dir : string
            Folder where to save the tiles and masks.
        """

        step_x = tile_width - overlap_x
        step_y = tile_height - overlap_y

        tile_count = 0
        for y in range(0, self.size[1] - tile_height + 1, step_y):
            for x in range(0, self.size[0] - tile_width + 1, step_x):
                box = (x, y, x + tile_width, y + tile_height)
                tile = self.rgb.crop(box)
                tile.save(os.path.join(output_dir, f"{tiles_name}_{tile_count}.png"))
                
                mask_array = self.mask[y:y + tile_height, x:x + tile_width]
                mask_image = Image.fromarray(mask_array)
                mask_image.save(os.path.join(output_dir, f"{tiles_name}_mask_{tile_count}.png"))

                tile_count += 1



# Class for creating the dialog to set up the parameters for the tool pick color
class PickColorDialog(tk.Toplevel):
    def __init__(self, parent, callback, window_title, initial_values=None):
        """Constructor.

        Parameters
        ----------
        parent : TK class
            Parent object.
        callback : function
            Routine to call to return parameter values to the caller.
        window_title : string
            Title of the window.
        initial_values : dictionary, optional
            Initialisation value for the parameters, by default None
        """
        super().__init__(parent)
        self.title(window_title)
        self.geometry("300x500")
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
        """Routine to create the dialogue."""
        # Radio buttons
        mode_frame = ttk.LabelFrame(self, text="Mode")
        mode_frame.pack(pady=10)
        hsv_radio = ttk.Radiobutton(mode_frame, text="HSV", variable=self.mode, value="HSV", command=self.update_mode)
        rgb_radio = ttk.Radiobutton(mode_frame, text="RGB", variable=self.mode, value="RGB", command=self.update_mode)
        hsv_radio.grid(row=0, column=0, padx=10)
        rgb_radio.grid(row=0, column=1, padx=10)

        # Frame to hold both HSV and RGB columns
        dual_column_frame = ttk.Frame(self)
        dual_column_frame.pack(pady=10)

        # HSV column
        hsv_column = ttk.Frame(dual_column_frame)
        hsv_column.grid(row=0, column=0, padx=(0, 10))
        for param in ['H', 'S', 'V']:
            frame = ttk.Frame(hsv_column)
            frame.pack(pady=5)
            label = ttk.Label(frame, text=self.symbol_label[param])
            label.grid(row=0, column=0, padx=5)
            minus_btn = ttk.Button(frame, text="-", width=3, command=lambda p=param: self.update_value(p, -1))
            minus_btn.grid(row=0, column=1)
            entry = ttk.Entry(frame, width=5, justify='center')
            entry.insert(0, str(self.params[param]['value']))
            entry.grid(row=0, column=2)
            plus_btn = ttk.Button(frame, text="+", width=3, command=lambda p=param: self.update_value(p, 1))
            plus_btn.grid(row=0, column=3)
            self.entries[param] = entry
            self.buttons[param] = (minus_btn, plus_btn)

        # Vertical separator
        ttk.Separator(dual_column_frame, orient='vertical').grid(row=0, column=1, sticky='ns', padx=5)

        # RGB column
        rgb_column = ttk.Frame(dual_column_frame)
        rgb_column.grid(row=0, column=2, padx=(10, 0))
        for param in ['R', 'G', 'B']:
            frame = ttk.Frame(rgb_column)
            frame.pack(pady=5)
            label = ttk.Label(frame, text=self.symbol_label[param])
            label.grid(row=0, column=0, padx=5)
            minus_btn = ttk.Button(frame, text="-", width=3, command=lambda p=param: self.update_value(p, -1))
            minus_btn.grid(row=0, column=1)
            entry = ttk.Entry(frame, width=5, justify='center')
            entry.insert(0, str(self.params[param]['value']))
            entry.grid(row=0, column=2)
            plus_btn = ttk.Button(frame, text="+", width=3, command=lambda p=param: self.update_value(p, 1))
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
                    messagebox.showerror("Invalid Input", f"Please enter a valid value for {param} between {self.params[param]['min']} and {self.params[param]['max']}")
                    return
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid value for {param} between {self.params[param]['min']} and {self.params[param]['max']}")
                return
           
                
        selected_mode = self.mode.get()
        self.callback(values, selected_mode)  # Pass both values and mode
        self.destroy()
        
    def cancel(self):
        """Routine to cancel the dialogue."""
        self.callback(None, None)
        self.destroy()
        
# Class for creating the dialog to set up the parameters for finding the regions
class FindRegionsDialog(tk.Toplevel):
    def __init__(self, parent, callback, project_data, initial_values=None, overwrite_flag = False):
        """Constructor.

        Parameters
        ----------
        parent : TK class
            Parent object.
        callback : function
            Routine to call to return parameter values to the caller.
        project_data : object
            Project data object.
        initial_values : dictionary, optional
            Initialisation value for the parameters, by default None
        overwrite_flag : logical, optional
            Flag to let the regions overwrite existing mask, by default False
        """
        super().__init__(parent)
        self.title("Find Regions Dialogue")
        self.geometry("400x600")
        self.resizable(False, False)
        self.callback = callback
        self.project_data = project_data
        self.overwrite_flag = overwrite_flag

        # Parameter names and range
        self.params = {
            'min H': {'value': 0, 'min': 0, 'max': 179},
            'min S': {'value': 0, 'min': 0, 'max': 255},
            'min V': {'value': 0, 'min': 0, 'max': 255},
            'min R': {'value': 0, 'min': 0, 'max': 255},
            'min G': {'value': 0, 'min': 0, 'max': 255},
            'min B': {'value': 0, 'min': 0, 'max': 255},
            'max H': {'value': 0, 'min': 0, 'max': 179},
            'max S': {'value': 0, 'min': 0, 'max': 255},
            'max V': {'value': 0, 'min': 0, 'max': 255},
            'max R': {'value': 0, 'min': 0, 'max': 255},
            'max G': {'value': 0, 'min': 0, 'max': 255},
            'max B': {'value': 0, 'min': 0, 'max': 255}
        }
        
        self.units = ["px"]

        if initial_values is not None:
            if "units" in initial_values:
                self.units = initial_values["units"]
        
        self.mode = tk.StringVar(value='HSV')
        self.edge_threshold1 = tk.IntVar(value=1)
        self.edge_threshold2 = tk.IntVar(value=1000)
        self.area_min = tk.DoubleVar(value=1)
        self.area_max = tk.DoubleVar(value=1000)
        self.ratio_threshold = tk.DoubleVar(value=0.8)
        self.area_unit = tk.StringVar(value="px")  # Default unit
        # Override with initial values if provided
        if initial_values:
            for key in self.params:
                if key in initial_values:
                    self.params[key]['value'] = initial_values[key]
            if 'mode' in initial_values:
                self.mode = tk.StringVar(value=initial_values['mode'])
            if 'edge_threshold1' in initial_values:
                self.edge_threshold1 = tk.IntVar(value=initial_values['edge_threshold1'])
            if 'edge_threshold2' in initial_values:
                self.edge_threshold2 = tk.IntVar(value=initial_values['edge_threshold2'])
            if 'area_min' in initial_values:
                self.area_min = tk.DoubleVar(value=initial_values['area_min'])
            if 'area_max' in initial_values:
                self.area_max = tk.DoubleVar(value=initial_values['area_max'])
            if 'area_unit' in initial_values:
                self.area_unit = tk.StringVar(value=initial_values['area_unit'])
            if 'ratio_threshold' in initial_values:
                self.ratio_threshold = tk.DoubleVar(value=initial_values['ratio_threshold'])

        self.entries = {}
        self.buttons = {}
        
        # Create a canvas and a vertical scrollbar for scrolling
        container = ttk.Frame(self)
        container.pack(fill='both', expand=True)

        canvas = tk.Canvas(container, borderwidth=0, height=350)
        self.frame = ttk.Frame(canvas)


        canvas.create_window((0, 0), window=self.frame, anchor="nw")



        canvas.pack(side="left", fill="both", expand=True)

        
        self.create_widgets()

    def create_widgets(self):
        """Routine to create the dialogue.
        """

        # Edge thresholds section
        edge_frame = ttk.LabelFrame(self.frame, text="Edge thresholds")
        edge_frame.pack(pady=10)

        self.edge_min = self.edge_threshold1
        self.edge_max = self.edge_threshold2

        min_label = ttk.Label(edge_frame, text="Min:")
        min_label.grid(row=0, column=0, padx=5)

        min_entry = ttk.Entry(edge_frame, textvariable=self.edge_min, width=6, justify='center')
        min_entry.grid(row=0, column=1, padx=5)

        max_label = ttk.Label(edge_frame, text="Max:")
        max_label.grid(row=0, column=2, padx=5)

        max_entry = ttk.Entry(edge_frame, textvariable=self.edge_max, width=6, justify='center')
        max_entry.grid(row=0, column=3, padx=5)
        
        preview_edge_btn = ttk.Button(edge_frame, text="Preview", command=self.preview_edge)
        preview_edge_btn.grid(row=0, column=4, padx=5)
        
        # Area thresholds section
        area_frame = ttk.LabelFrame(self.frame, text="Area thresholds")
        area_frame.pack(pady=10)

        area_min_label = ttk.Label(area_frame, text="Min:")
        area_min_label.grid(row=0, column=0, padx=5)

        area_min_entry = ttk.Entry(area_frame, textvariable=self.area_min, width=8, justify='center')
        area_min_entry.grid(row=0, column=1, padx=5)

        area_max_label = ttk.Label(area_frame, text="Max:")
        area_max_label.grid(row=0, column=2, padx=5)

        area_max_entry = ttk.Entry(area_frame, textvariable=self.area_max, width=8, justify='center')
        area_max_entry.grid(row=0, column=3, padx=5)
        
        
        unit_label = ttk.Label(area_frame, text="Unit:")
        unit_label.grid(row=0, column=4, padx=5)

        unit_combo = ttk.Combobox(area_frame, textvariable=self.area_unit, values=self.units, state="readonly", width=6)
        unit_combo.grid(row=0, column=5, padx=5)


        # Ratio threshold section
        ratio_frame = ttk.LabelFrame(self.frame, text="Ratio thresholds")
        ratio_frame.pack(pady=10)

        ratio_label = ttk.Label(ratio_frame, text="Ratio Threshold:")
        ratio_label.grid(row=0, column=0, padx=5)

        ratio_entry = ttk.Entry(ratio_frame, textvariable=self.ratio_threshold, width=10, justify='center')
        ratio_entry.grid(row=0, column=1, padx=5)

        # Radio buttons
        mode_frame = ttk.LabelFrame(self.frame, text="Mode")
        mode_frame.pack(pady=10)

        hsv_radio = ttk.Radiobutton(mode_frame, text="HSV", variable=self.mode, value="HSV", command=self.update_mode)
        rgb_radio = ttk.Radiobutton(mode_frame, text="RGB", variable=self.mode, value="RGB", command=self.update_mode)
        hsv_radio.grid(row=0, column=0, padx=10)
        rgb_radio.grid(row=0, column=1, padx=10)

        # Frame to hold both HSV and RGB columns
        dual_column_frame = ttk.Frame(self.frame)
        dual_column_frame.pack(pady=10)

        # HSV column
        hsv_column = ttk.Frame(dual_column_frame)
        hsv_column.grid(row=0, column=0, padx=(0, 0))
        for param in ['min H', 'max H', 'min S', 'max S', 'min V', 'max V']:
            frame = ttk.Frame(hsv_column)
            frame.pack(pady=5)

            label = ttk.Label(frame, text=param)

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
            
        # Vertical separator
        ttk.Separator(dual_column_frame, orient='vertical').grid(row=0, column=1, sticky='ns', padx=5)
        
        # RGB column
        rgb_column = ttk.Frame(dual_column_frame)
        rgb_column.grid(row=0, column=2, padx=(10, 0))
        for param in ['min R', 'max R', 'min G', 'max G', 'min B', 'max B']:
            frame = ttk.Frame(rgb_column)
            frame.pack(pady=5)

            label = ttk.Label(frame, text=param)

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
            
        preview_colors_btn = ttk.Button(dual_column_frame, text="Preview", command=self.preview_colors)
        preview_colors_btn.grid(row=1, column=1, padx=5)

        self.update_mode()  # Set initial state
        
        

        
        # Submit and Cancel buttons
        btn_frame = ttk.Frame(self.frame)
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
            active = (mode == "HSV" and param in ('min H', 'max H', 'min S', 'max S', 'min V', 'max V')) or (mode == "RGB" and param in ('min R', 'max R', 'min G', 'max G', 'min B', 'max B'))
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
                    messagebox.showerror("Invalid Input", f"Please enter a valid value for {param} between {self.params[param]['min']} and {self.params[param]['max']}")
                    return
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid value for {param} between {self.params[param]['min']} and {self.params[param]['max']}")
                return
        for param in ['H', 'S', 'V', 'R', 'G', 'B']:
            try:
                val = int(self.entries['min ' + param].get())
                
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid value for {'min ' + param}")
                return
            try:
                val = int(self.entries['max ' + param].get())
                
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid value for {'max ' + param}")
                return
            if self.params['min ' + param]['value'] > self.params['max ' + param]['value']:
                messagebox.showerror("Invalid Input", f"The minimum value for {param} cannot be greater than the maximum value for {param}")
                return
        try:
            val = int(self.edge_min.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a valid value for the minimum threshold value for edges")
            return
        try:
            val = int(self.edge_max.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a valid value for the maximum threshold value for edges")
            return
        if self.edge_min.get() > self.edge_max.get():
                messagebox.showerror("Invalid Input", f"The minimum threshold value for edges cannot be greater than the maximum threshold value")
                return
        try:
            val = int(self.area_min.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a valid value for the minimum area")
            return
        try:
            val = int(self.area_max.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a valid value for the maximum area")
            return
        if self.area_min.get() > self.area_max.get():
                messagebox.showerror("Invalid Input", f"The minimum area value cannot be greater than the maximum area value")
                return
        
        try:
            val = float(self.ratio_threshold.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a valid value for the ratio threshold")
            return
        
        values['edge_min'] = self.edge_min.get()
        values['edge_max'] = self.edge_max.get()
        
        
        values['area_min'] = self.area_min.get()
        values['area_max'] = self.area_max.get()
        values['area_unit'] = self.area_unit.get()
        
        values['ratio_threshold'] = self.ratio_threshold.get()


        selected_mode = self.mode.get()
        self.callback(values, selected_mode)  # Pass both values and mode
        self.destroy()

    def validate(self):
        """Routine to validate inputs.
        
         Returns
        -------
        logical
            Check if inputs are fine.
        """
        for param in self.params:
            try:
                val = int(self.entries[param].get())
                if not(self.params[param]['min'] <= val <= self.params[param]['max']):
                    messagebox.showerror("Invalid Input", f"Please enter a valid value for {param} between {self.params[param]['min']} and {self.params[param]['max']}")
                    return 1
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid value for {param} between {self.params[param]['min']} and {self.params[param]['max']}")
                return False
        for param in ['H', 'S', 'V', 'R', 'G', 'B']:
            try:
                val = int(self.entries['min ' + param].get())
                
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid value for {'min ' + param}")
                return False
            try:
                val = int(self.entries['max ' + param].get())
                
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid value for {'max ' + param}")
                return False
            if self.params['min ' + param]['value'] > self.params['max ' + param]['value']:
                messagebox.showerror("Invalid Input", f"The minimum value for {param} cannot be greater than the maximum value for {param}")
                return False
        try:
            val = int(self.edge_min.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a valid value for the minimum threshold value for edges")
            return False
        try:
            val = int(self.edge_max.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a valid value for the maximum threshold value for edges")
            return False
        if self.edge_min.get() > self.edge_max.get():
                messagebox.showerror("Invalid Input", f"The minimum threshold value for edges cannot be greater than the maximum threshold value")
                return False
        try:
            val = int(self.area_min.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a valid value for the minimum area")
            return False
        try:
            val = int(self.area_max.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a valid value for the maximum area")
            return False
        if self.area_min.get() > self.area_max.get():
                messagebox.showerror("Invalid Input", f"The minimum area value cannot be greater than the maximum area value")
                return False
        
        try:
            val = float(self.ratio_threshold.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a valid value for the ratio threshold")
            return False
        return True
        
        
    def cancel(self):
        """Routine to cancel the dialogue."""
        self.callback(None, None)
        self.destroy() 
        
    def preview_edge(self):
        """Routine to preview the detected edges."""
        
        if self.validate(): 
            edges = self.project_data.find_regions(None, None, "RGB", self.edge_min.get(),
                        self.edge_max.get(), None, None,
                        None , preview="edges", overwrite = self.overwrite_flag)
            cv2.imshow('edges',edges)
            cv2.waitKey(0)
            
    def preview_colors(self):
        """Routine to preview the selected colors."""
        
        if self.validate(): 
            if self.mode.get() == 'HSV':
                range_color = [[int(self.entries['min H'].get()), int(self.entries['min S'].get()), int(self.entries['min V'].get())],
                               [int(self.entries['max H'].get()), int(self.entries['max S'].get()), int(self.entries['max V'].get())]]
            else:
                range_color = [[int(self.entries['min R'].get()), int(self.entries['min G'].get()), int(self.entries['min B'].get())],
                               [int(self.entries['max R'].get()), int(self.entries['max G'].get()), int(self.entries['max B'].get())]]
            colors = self.project_data.find_regions(None, range_color, self.mode.get(), None,
                        None, None, None,
                        None , preview="colors", overwrite = self.overwrite_flag)
            cv2.imshow('colors',colors)
            cv2.waitKey(0)
        


# Class for creating the dialog to select a range of colours
class ColorRangeDialog(tk.Toplevel):
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
        self.title("Color Range Dialogue")
        self.geometry("400x600")
        self.resizable(False, False)
        self.callback = callback

        # Parameter names and range
        self.params = {
            'min H': {'value': 0, 'min': 0, 'max': 179},
            'min S': {'value': 0, 'min': 0, 'max': 255},
            'min V': {'value': 0, 'min': 0, 'max': 255},
            'min R': {'value': 0, 'min': 0, 'max': 255},
            'min G': {'value': 0, 'min': 0, 'max': 255},
            'min B': {'value': 0, 'min': 0, 'max': 255},
            'max H': {'value': 0, 'min': 0, 'max': 179},
            'max S': {'value': 0, 'min': 0, 'max': 255},
            'max V': {'value': 0, 'min': 0, 'max': 255},
            'max R': {'value': 0, 'min': 0, 'max': 255},
            'max G': {'value': 0, 'min': 0, 'max': 255},
            'max B': {'value': 0, 'min': 0, 'max': 255}
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
        
        # Create a canvas and a vertical scrollbar for scrolling
        container = ttk.Frame(self)
        container.pack(fill='both', expand=True)

        canvas = tk.Canvas(container, borderwidth=0, height=350)
        self.frame = ttk.Frame(canvas)


        canvas.create_window((0, 0), window=self.frame, anchor="nw")



        canvas.pack(side="left", fill="both", expand=True)

        
        self.create_widgets()

    def create_widgets(self):
        """Routine to create the dialogue.
        """

        # Radio buttons
        mode_frame = ttk.LabelFrame(self.frame, text="Mode")
        mode_frame.pack(pady=10)

        hsv_radio = ttk.Radiobutton(mode_frame, text="HSV", variable=self.mode, value="HSV", command=self.update_mode)
        rgb_radio = ttk.Radiobutton(mode_frame, text="RGB", variable=self.mode, value="RGB", command=self.update_mode)
        hsv_radio.grid(row=0, column=0, padx=10)
        rgb_radio.grid(row=0, column=1, padx=10)

        # Frame to hold both HSV and RGB columns
        dual_column_frame = ttk.Frame(self.frame)
        dual_column_frame.pack(pady=10)

        # HSV column
        hsv_column = ttk.Frame(dual_column_frame)
        hsv_column.grid(row=0, column=0, padx=(0, 0))
        for param in ['min H', 'max H', 'min S', 'max S', 'min V', 'max V']:
            frame = ttk.Frame(hsv_column)
            frame.pack(pady=5)

            label = ttk.Label(frame, text=param)

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
            
        # Vertical separator
        ttk.Separator(dual_column_frame, orient='vertical').grid(row=0, column=1, sticky='ns', padx=5)
        
        # RGB column
        rgb_column = ttk.Frame(dual_column_frame)
        rgb_column.grid(row=0, column=2, padx=(10, 0))
        for param in ['min R', 'max R', 'min G', 'max G', 'min B', 'max B']:
            frame = ttk.Frame(rgb_column)
            frame.pack(pady=5)

            label = ttk.Label(frame, text=param)

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
        btn_frame = ttk.Frame(self.frame)
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
            active = (mode == "HSV" and param in ('min H', 'max H', 'min S', 'max S', 'min V', 'max V')) or (mode == "RGB" and param in ('min R', 'max R', 'min G', 'max G', 'min B', 'max B'))
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
                    messagebox.showerror("Invalid Input", f"Please enter a valid value for {param} between {self.params[param]['min']} and {self.params[param]['max']}")
                    return
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid value for {param} between {self.params[param]['min']} and {self.params[param]['max']}")
                return
        for param in ['H', 'S', 'V', 'R', 'G', 'B']:
            try:
                val = int(self.entries['min ' + param].get())
                
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid value for {'min ' + param}")
                return
            try:
                val = int(self.entries['max ' + param].get())
                
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid value for {'max ' + param}")
                return
            if self.params['min ' + param]['value'] > self.params['max ' + param]['value']:
                messagebox.showerror("Invalid Input", f"The minimum value for {param} cannot be greater than the maximum value for {param}")
                return
        

        selected_mode = self.mode.get()
        self.callback(values, selected_mode)  # Pass both values and mode
        self.destroy()
        
    def cancel(self):
        """Routine to cancel the dialogue."""
        self.callback(None, None)
        self.destroy()      
 
# Class for the popup window to check the consistency of the mask        
class CustomPopupMaskConsistency:
    def __init__(self, parent):
        """Constructor.

        Parameters
        ----------
        parent : TK class
            Parent object.
        """
        self.choice = None
        self.top = tk.Toplevel(parent)
        self.top.title("Mask consistency")

        # Message
        label = tk.Label(self.top, text="Not all pixel in the mask are assigned to classes. What class should be used for the not assigned pixels?", font=("Arial", 14)).pack(pady=10)

        
        # Frame to hold buttons horizontally
        button_frame = tk.Frame(self.top)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        
        # Create buttons in a row
        for label in ["Background", "Matrix", "Inclusions", "Cancel"]:
            tk.Button(button_frame, text=label, width=12,
                      command=lambda l=label: self.set_choice(l)).pack(side=tk.LEFT, padx=5)


        # Wait for the window to close
        self.top.grab_set()
        parent.wait_window(self.top)

    def set_choice(self, label):
        """Routine to access the pressed button.
        
        Parameters
        ----------
        label : string
            Label of the pressed button.
        """
        self.choice = label
        self.top.destroy()
  
        
# Class implementing the app
class PetroSeg:
    def __init__(self, root):
        """Constructor for the app.

        Parameters
        ----------
        root : Tkinter root
            Root
        """
        
        
        self.root = root
        self.root.title("PetroSeg")
        self.root.geometry("512x512")
        
        
        # Get the path of the current script
        script_dir = Path(__file__).resolve().parent
        
        
        # Build the path to splash.png inside the assets folder
        splash_path = script_dir / "assets" / "splash.png"


        self.show_splash(root, splash_path, duration=3000)

        self.zoom_level = 1.0 # Level of Zoom
        self.project = ProjectData(console_debug = CONSOLE_DEBUG)

        # Init initial status app
        self.status = STATUS_NONE

        # Init initial status tools
        self.status_tool = STATUS_TOOL_NONE

        # Fields for cropping the image and selecting a rectangular region
        self.start_x = None
        self.start_y = None
        self.rect_id = None

        # Variable for the display
        self.display_status = STATUS_DISPLAY_NONE
        
        # Selected color to use on the mask
        self.current_mask_color = COLOR_MASK_NONE
        
        # Tolerance on HSV and RGB values for the tool pick color
        self.pick_color_params = {
            'H': 10,
            'S': 10,
            'V': 10,
            'R': 10,
            'G': 10,
            'B': 10,
            "mode": 'HSV'
        }

        # Tolerance on HSV and RGB values for the tool pick region
        self.pick_region_params = {
            'H': 10,
            'S': 10,
            'V': 10,
            'R': 10,
            'G': 10,
            'B': 10,
            "mode": 'HSV'
        }

        # Data for the find regions dialogue
        self.find_regions_params = {
            'min H': 0,
            'min S': 0,
            'min V': 0,
            'min R': 0,
            'min G': 0,
            'min B': 0,
            'max H': 0,
            'max S': 0,
            'max V': 0,
            'max R': 0,
            'max G': 0,
            'max B': 0,
            "mode": 'HSV',
            'edge_threshold1': 100,
            'edge_threshold2': 200,
            'area_min': 1,
            'area_max': 1000,
            'area_min_px': 0,
            'area_max_px': 0,
            'area_unit': 'px',
            'ratio_threshold': 0.8,
            'units' : []
        }
        
        # Data for the color range dialogue
        self.color_range_params = {
            'min H': 0,
            'min S': 0,
            'min V': 0,
            'min R': 0,
            'min G': 0,
            'min B': 0,
            'max H': 0,
            'max S': 0,
            'max V': 0,
            'max R': 0,
            'max G': 0,
            'max B': 0,
            "mode": 'HSV'
        }
        
        # Variables to drag the image with the mouse
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Variables for the ruler
        self.start_point = None
        self.temp_line = None
        self.final_line = None
        self.temp_text = None
        self.markers = []
        

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
        self.pick_region_flag = tk.BooleanVar(value=False)
        self.unpick_region_flag = tk.BooleanVar(value=False)
        self.ruler_flag = tk.BooleanVar(value=False)
        self.pick_rect_flag = tk.BooleanVar(value=False)
        self.unpick_rect_flag = tk.BooleanVar(value=False)
        self.overwrite_flag = tk.BooleanVar(value=False)
        self.image_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.image_menu.add_command(label="Export Image...", command=self.export_image)
        self.image_menu.add_command(label="Create Tiles...", command=self.create_tiles)
        self.image_menu.add_separator()
        self.image_menu.add_checkbutton(label="Crop", command=self.crop_tool, variable=self.crop_flag)
        self.image_menu.add_checkbutton(label="Pick Color", command=self.pick_color_tool, variable=self.pick_color_flag)
        self.image_menu.add_checkbutton(label="Unpick Color", command=self.unpick_color_tool, variable=self.unpick_color_flag)
        self.image_menu.add_checkbutton(label="Pick Region", command=self.pick_region_tool, variable=self.pick_region_flag)
        self.image_menu.add_checkbutton(label="Unpick Region", command=self.unpick_region_tool, variable=self.unpick_region_flag)
        self.image_menu.add_command(label="Pick Color Range...", command=self.open_color_range_dialog)
        self.image_menu.add_checkbutton(label="Pick Rect", command=self.pick_rect_tool, variable=self.pick_rect_flag)
        self.image_menu.add_checkbutton(label="Unpick Rect", command=self.unpick_rect_tool, variable=self.unpick_rect_flag)
        self.image_menu.add_checkbutton(label="Ruler", command=self.ruler_tool, variable=self.ruler_flag)
        self.image_menu.add_separator()
        self.image_menu.add_command(label="Pick Color Dialog...", command=self.open_pick_color_dialog)
        self.image_menu.add_command(label="Pick Region Dialog...", command=self.open_pick_region_dialog)
        self.image_menu.add_command(label="Set Ruler...", command=self.open_set_ruler_dialog)
        self.image_menu.add_command(label="Find Regions...", command=self.open_find_regions_dialog)
        self.image_menu.add_separator()
        self.image_menu.add_command(label="Show H Histogram", command=self.show_h_histogram)
        self.image_menu.add_command(label="Show S Histogram", command=self.show_s_histogram)
        self.image_menu.add_command(label="Show V Histogram", command=self.show_v_histogram)
        self.image_menu.add_command(label="Show R Histogram", command=self.show_r_histogram)
        self.image_menu.add_command(label="Show G Histogram", command=self.show_g_histogram)
        self.image_menu.add_command(label="Show B Histogram", command=self.show_b_histogram)
        self.menu_bar.add_cascade(label="Image", menu=self.image_menu)
        # Mask Menu
        self.background_flag = tk.BooleanVar(value=False)
        self.matrix_flag = tk.BooleanVar(value=False)
        self.inclusion_flag = tk.BooleanVar(value=False)
        self.mask_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.mask_menu.add_checkbutton(label="Export Mask...", command=self.export_mask)
        self.mask_menu.add_checkbutton(label="Export Mix...", command=self.export_mix)
        self.mask_menu.add_separator()
        self.mask_menu.add_checkbutton(label="Background", command=self.set_background, variable=self.background_flag)
        self.mask_menu.add_checkbutton(label="Matrix", command=self.set_matrix, variable=self.matrix_flag)
        self.mask_menu.add_checkbutton(label="Inclusion", command=self.set_inclusion, variable=self.inclusion_flag)
        self.mask_menu.add_checkbutton(label="Overwrite", variable=self.overwrite_flag)
        self.mask_menu.add_command(label="Check Consistency...", command=self.check_consistency)
        self.mask_menu.add_separator()
        self.mask_menu.add_command(label="Export Statistics...", command=self.export_statistics)
        self.menu_bar.add_cascade(label="Mask", menu=self.mask_menu)
        # Help Menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="About", command=lambda: self.show_about_window(root, "dice.jpeg"))
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        # Attach the menu bar to the root window
        root.config(menu=self.menu_bar)

        # Init entries in menus
        self.file_menu.entryconfig("Save Project", state="disabled")
        self.image_menu.entryconfig("Export Image...", state="disabled")
        self.image_menu.entryconfig("Create Tiles...", state="disabled")
        self.edit_menu.entryconfig("Zoom in", state="disabled")
        self.edit_menu.entryconfig("Zoom out", state="disabled")
        self.edit_menu.entryconfig("Undo", state="disabled")
        self.edit_menu.entryconfig("Redo", state="disabled")
        self.image_menu.entryconfig("Crop", state="disabled")
        self.image_menu.entryconfig("Pick Color", state="disabled")
        self.image_menu.entryconfig("Unpick Color", state="disabled")
        self.image_menu.entryconfig("Pick Color Dialog...", state="disabled")
        self.image_menu.entryconfig("Pick Region", state="disabled")
        self.image_menu.entryconfig("Unpick Region", state="disabled")
        self.image_menu.entryconfig("Pick Region Dialog...", state="disabled")
        self.image_menu.entryconfig("Pick Color Range...", state="disabled")
        self.image_menu.entryconfig("Pick Rect", state="disabled")
        self.image_menu.entryconfig("Unpick Rect", state="disabled")
        self.image_menu.entryconfig("Set Ruler...", state="disabled")
        self.image_menu.entryconfig("Find Regions...", state="disabled")
        self.image_menu.entryconfig("Ruler", state="disabled")
        self.image_menu.entryconfig("Show H Histogram", state="disabled")
        self.image_menu.entryconfig("Show S Histogram", state="disabled")
        self.image_menu.entryconfig("Show V Histogram", state="disabled")
        self.image_menu.entryconfig("Show R Histogram", state="disabled")
        self.image_menu.entryconfig("Show G Histogram", state="disabled")
        self.image_menu.entryconfig("Show B Histogram", state="disabled")
        self.mask_menu.entryconfig("Background", state="disabled")
        self.mask_menu.entryconfig("Matrix", state="disabled")
        self.mask_menu.entryconfig("Inclusion", state="disabled")
        self.mask_menu.entryconfig("Overwrite", state="disabled")
        self.mask_menu.entryconfig("Check Consistency...", state="disabled")
        self.mask_menu.entryconfig("Export Statistics...", state="disabled")
        
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
        self.canvas.configure(yscrollincrement='2')
        self.canvas.configure(xscrollincrement='2')
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
        self.canvas.bind("<Motion>", self.on_motion)
        
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
                    self.project = ProjectData(console_debug = CONSOLE_DEBUG)
                else:
                    return
            elif result is False:
                self.project = ProjectData(console_debug = CONSOLE_DEBUG)
        
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.canvas.config(cursor="watch")
            self.project = ProjectData(image_path = file_path, console_debug = CONSOLE_DEBUG)
            self.change_status(STATUS_IMAGE_LOADED)
            self.zoom_level = 1.0
            self.change_status_display(STATUS_DISPLAY_IMAGE)
            self.display_image()
            self.update_image_size_status(self.project.size)
            self.change_status_tool(STATUS_TOOL_NONE)
            self.change_status_display(STATUS_DISPLAY_IMAGE)
            self.canvas.config(cursor="arrow")
            
    
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
                if self.current_mask_color == COLOR_MASK_NONE:
                    messagebox.showerror("Error", "No mask class selected.")
                    return
                canvas_x = self.canvas.canvasx(event.x)
                canvas_y = self.canvas.canvasy(event.y)
                x, y = int(canvas_x / self.zoom_level), int(canvas_y / self.zoom_level)
                width, height = self.project.size
                if 0 <= x < width and 0 <= y < height:
                    if self.pick_color_params['mode'] == 'HSV':
                        # Get the color at the clicked point
                        hue = self.project.hsv[y, x, 0]
                        lower = np.array([0, 0, 0])
                        upper = np.array([0, 0, 0])
                        if hue > self.pick_color_params['H']:
                            lower[0] = hue - self.pick_color_params['H']
                        else:
                            lower[0] = 0
                        if hue < 179 - self.pick_color_params['H']:
                            upper[0] = hue + self.pick_color_params['H']
                        else:
                            upper[0] = 179
                            
                        sat = self.project.hsv[y, x, 1]
                        if sat > self.pick_color_params['S']:
                            lower[1] = sat - self.pick_color_params['S']
                        else:
                            lower[1] = 0
                        if sat < 255 - self.pick_color_params['S']:
                            upper[1] = sat + self.pick_color_params['S']
                        else:
                            upper[1] = 255
                            
                        bri = self.project.hsv[y, x, 2]
                        if bri > self.pick_color_params['V']:
                            lower[2] = bri - self.pick_color_params['V']
                        else:
                            lower[2] = 0
                        if bri < 255 - self.pick_color_params['V']:
                            upper[2] = bri + self.pick_color_params['V']
                        else:
                            upper[2] = 255
                        self.canvas.config(cursor="watch")
                        self.project.select_color_hsv(lower, upper, self.current_mask_color, overwrite = self.overwrite_flag.get())
                        self.canvas.config(cursor="cross")
                    elif self.pick_color_params['mode'] == 'RGB':
                        # Get the color at the clicked point
                        red, green, blue = self.project.rgb.getpixel((x, y))
                        lower = np.array([0, 0, 0])
                        upper = np.array([0, 0, 0])
                        if red > self.pick_color_params['R']:
                            lower[0] = red - self.pick_color_params['R']
                        else:
                            lower[0] = 0
                        if red < 255 - self.pick_color_params['R']:
                            upper[0] = red + self.pick_color_params['R']
                        else:
                            upper[0] = 255
                            
                        if green > self.pick_color_params['G']:
                            lower[1] = green - self.pick_color_params['G']
                        else:
                            lower[1] = 0
                        if green < 255 - self.pick_color_params['G']:
                            upper[1] = green + self.pick_color_params['G']
                        else:
                            upper[1] = 255
                            
                        if blue > self.pick_color_params['B']:
                            lower[2] = blue - self.pick_color_params['B']
                        else:
                            lower[2] = 0
                        if blue < 255 - self.pick_color_params['B']:
                            upper[2] = blue + self.pick_color_params['B']
                        else:
                            upper[2] = 255
                        self.canvas.config(cursor="watch")
                        self.project.select_color_rgb(lower, upper, self.current_mask_color, overwrite = self.overwrite_flag.get())
                        self.canvas.config(cursor="cross")
                    self.update_undo_redo()
                    self.file_menu.entryconfig("Save Project", state="normal")
                    self.display_image()
            # Code to pick a region
            elif self.status_tool == STATUS_TOOL_PICK_REGION:
                if self.current_mask_color == COLOR_MASK_NONE:
                    messagebox.showerror("Error", "No mask class selected.")
                    return
                canvas_x = self.canvas.canvasx(event.x)
                canvas_y = self.canvas.canvasy(event.y)
                x, y = int(canvas_x / self.zoom_level), int(canvas_y / self.zoom_level)
                width, height = self.project.size
                if 0 <= x < width and 0 <= y < height:
                    if self.pick_region_params['mode'] == 'HSV':
                        lower = (self.pick_region_params['H'], self.pick_region_params['S'], self.pick_region_params['V'])
                        upper = (self.pick_region_params['H'], self.pick_region_params['S'], self.pick_region_params['V'])
                        self.canvas.config(cursor="watch")
                        self.project.select_region_hsv((x, y), lower, upper, self.current_mask_color, overwrite = self.overwrite_flag.get())
                        self.canvas.config(cursor="cross")
                    elif self.pick_region_params['mode'] == 'RGB':
                        lower = (self.pick_region_params['R'], self.pick_region_params['G'], self.pick_region_params['B'])
                        upper = (self.pick_region_params['R'], self.pick_region_params['G'], self.pick_region_params['B'])
                        self.canvas.config(cursor="watch")
                        self.project.select_region_rgb((x, y), lower, upper, self.current_mask_color, overwrite = self.overwrite_flag.get())
                        self.canvas.config(cursor="cross")
                    self.update_undo_redo()
                    self.file_menu.entryconfig("Save Project", state="normal")
                    self.display_image()
            # Code to unpick a region
            elif self.status_tool == STATUS_TOOL_UNPICK_REGION:
                canvas_x = self.canvas.canvasx(event.x)
                canvas_y = self.canvas.canvasy(event.y)
                x, y = int(canvas_x / self.zoom_level), int(canvas_y / self.zoom_level)
                width, height = self.project.size
                if 0 <= x < width and 0 <= y < height:
                    self.canvas.config(cursor="watch")
                    self.project.deselect_region((x, y))
                    self.canvas.config(cursor="cross")
                    self.update_undo_redo()
                    self.file_menu.entryconfig("Save Project", state="normal")
                    self.display_image()
            # Code to unpick a color
            elif self.status_tool == STATUS_TOOL_UNPICK_COLOR:
                canvas_x = self.canvas.canvasx(event.x)
                canvas_y = self.canvas.canvasy(event.y)
                x, y = int(canvas_x / self.zoom_level), int(canvas_y / self.zoom_level)
                width, height = self.project.size
                if 0 <= x < width and 0 <= y < height:
                    if self.pick_color_params['mode'] == 'HSV':
                        # Get the color at the clicked point
                        hue = self.project.hsv[y, x, 0]
                        lower = np.array([0, 0, 0])
                        upper = np.array([0, 0, 0])
                        if hue > self.pick_color_params['H']:
                            lower[0] = hue - self.pick_color_params['H']
                        else:
                            lower[0] = 0
                        if hue < 179 - self.pick_color_params['H']:
                            upper[0] = hue + self.pick_color_params['H']
                        else:
                            upper[0] = 179
                            
                        sat = self.project.hsv[y, x, 1]
                        if sat > self.pick_color_params['S']:
                            lower[1] = sat - self.pick_color_params['S']
                        else:
                            lower[1] = 0
                        if sat < 255 - self.pick_color_params['S']:
                            upper[1] = sat + self.pick_color_params['S']
                        else:
                            upper[1] = 255
                            
                        bri = self.project.hsv[y, x, 2]
                        if bri > self.pick_color_params['V']:
                            lower[2] = bri - self.pick_color_params['V']
                        else:
                            lower[2] = 0
                        if bri < 255 - self.pick_color_params['V']:
                            upper[2] = bri + self.pick_color_params['V']
                        else:
                            upper[2] = 255
                        self.canvas.config(cursor="watch")
                        self.project.select_color_hsv(lower, upper, COLOR_MASK_NONE, overwrite = self.overwrite_flag.get())
                        self.canvas.config(cursor="cross")
                    elif self.pick_color_params['mode'] == 'RGB':
                        # Get the color at the clicked point
                        red, green, blue = self.project.rgb.getpixel((x, y))
                        lower = np.array([0, 0, 0])
                        upper = np.array([0, 0, 0])
                        if red > self.pick_color_params['R']:
                            lower[0] = red - self.pick_color_params['R']
                        else:
                            lower[0] = 0
                        if red < 255 - self.pick_color_params['R']:
                            upper[0] = red + self.pick_color_params['R']
                        else:
                            upper[0] = 255
                            
                        if green > self.pick_color_params['G']:
                            lower[1] = green - self.pick_color_params['G']
                        else:
                            lower[1] = 0
                        if green < 255 - self.pick_color_params['G']:
                            upper[1] = green + self.pick_color_params['G']
                        else:
                            upper[1] = 255
                            
                        if blue > self.pick_color_params['B']:
                            lower[2] = blue - self.pick_color_params['B']
                        else:
                            lower[2] = 0
                        if blue < 255 - self.pick_color_params['B']:
                            upper[2] = blue + self.pick_color_params['B']
                        else:
                            upper[2] = 255
                        self.canvas.config(cursor="watch")
                        self.project.select_color_rgb(lower, upper, COLOR_MASK_NONE, overwrite = self.overwrite_flag.get())
                        self.canvas.config(cursor="cross")
                    self.update_undo_redo()
                    self.file_menu.entryconfig("Save Project", state="normal")
                    self.display_image()
                    
    def on_motion(self, event):
        """Routine for actions on mouse motion"""

        self.show_info(event)
        
        if self.status_tool == STATUS_TOOL_RULER:
            self.on_drag_ruler(event)

    def show_info(self, event):
        """Routine to update the info on the left panel"""
        if self.status == STATUS_IMAGE_LOADED:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            x, y = int(canvas_x / self.zoom_level), int(canvas_y / self.zoom_level)
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
            else:
                raise ValueError("display_status not supported")

    def on_mouse_press(self, event):
        """Routine to capture a mouse press.

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
            elif self.status_tool == STATUS_TOOL_PICK_RECT:
                self.start_x = self.canvas.canvasx(event.x)
                self.start_y = self.canvas.canvasy(event.y)
                if self.rect_id:
                    self.canvas.delete(self.rect_id)
                self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")
            elif self.status_tool == STATUS_TOOL_UNPICK_RECT:
                self.start_x = self.canvas.canvasx(event.x)
                self.start_y = self.canvas.canvasy(event.y)
                if self.rect_id:
                    self.canvas.delete(self.rect_id)
                self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_mouse_drag(self, event):
        """Routine to capture the drag of the cursor.

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
            elif self.status_tool == STATUS_TOOL_RULER:    
                self.on_drag_ruler(self, event)
            elif self.status_tool == STATUS_TOOL_PICK_RECT:
                cur_x = self.canvas.canvasx(event.x)
                cur_y = self.canvas.canvasy(event.y)
                self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)
            elif self.status_tool == STATUS_TOOL_UNPICK_RECT:
                cur_x = self.canvas.canvasx(event.x)
                cur_y = self.canvas.canvasy(event.y)
                self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_mouse_release(self, event):
        """Routine to capture the release of the mouse botton.

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

                self.canvas.config(cursor="watch")
                self.project.crop_image(x1, y1, x2, y2)
                self.canvas.config(cursor="arrow")
                self.zoom_level = 1.0
                self.display_image()
                self.update_image_size_status(self.project.size)
                self.file_menu.entryconfig("Save Project", state="normal")
                self.update_undo_redo()
                self.change_status_tool(STATUS_TOOL_NONE)
            elif self.status_tool == STATUS_TOOL_PICK_RECT:
                if self.current_mask_color == COLOR_MASK_NONE:
                    messagebox.showerror("Error", "No mask class selected.")
                    return
                end_x = self.canvas.canvasx(event.x)
                end_y = self.canvas.canvasy(event.y)

                x1 = int(min(self.start_x, end_x) / self.zoom_level)
                y1 = int(min(self.start_y, end_y) / self.zoom_level)
                x2 = int(max(self.start_x, end_x) / self.zoom_level)
                y2 = int(max(self.start_y, end_y) / self.zoom_level)

                self.canvas.config(cursor="watch")
                self.project.mask_rectangle((x1, y1), (x2, y2), self.current_mask_color, overwrite = self.overwrite_flag.get())
                self.canvas.config(cursor="arrow")
                self.display_image()
                self.file_menu.entryconfig("Save Project", state="normal")
                self.update_undo_redo()
            elif self.status_tool == STATUS_TOOL_UNPICK_RECT:
                end_x = self.canvas.canvasx(event.x)
                end_y = self.canvas.canvasy(event.y)

                x1 = int(min(self.start_x, end_x) / self.zoom_level)
                y1 = int(min(self.start_y, end_y) / self.zoom_level)
                x2 = int(max(self.start_x, end_x) / self.zoom_level)
                y2 = int(max(self.start_y, end_y) / self.zoom_level)

                self.canvas.config(cursor="watch")
                self.project.unmask_rectangle((x1, y1), (x2, y2))
                self.canvas.config(cursor="arrow")
                self.display_image()
                self.file_menu.entryconfig("Save Project", state="normal")
                self.update_undo_redo()

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
                
    def export_mask(self):
        """Routine to save the current mask."""
        if self.status == STATUS_IMAGE_LOADED:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                   filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                image = Image.fromarray(self.project.mask)
                image.save(file_path)
                self.change_status_tool(STATUS_TOOL_NONE)
                
    def export_mix(self):
        """Routine to save the current mask overlaid to the current image."""
        if self.status == STATUS_IMAGE_LOADED:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                   filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                mask_rgb = Image.fromarray(self.project.mask)
                mix_rgb = Image.blend(self.project.rgb.convert("RGBA"), mask_rgb.convert("RGBA"), 0.5)
                mix_rgb.save(file_path)
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
                self.image_menu.entryconfig("Create Tiles...", state="normal")
                self.image_menu.entryconfig("Pick Color Dialog...", state="normal")
                self.image_menu.entryconfig("Pick Region", state="normal")
                self.image_menu.entryconfig("Unpick Region", state="normal")
                self.image_menu.entryconfig("Pick Region Dialog...", state="normal")
                self.image_menu.entryconfig("Pick Color Range...", state="normal")
                self.image_menu.entryconfig("Pick Rect", state="normal")
                self.image_menu.entryconfig("Unpick Rect", state="normal")
                self.image_menu.entryconfig("Set Ruler...", state="normal")
                self.image_menu.entryconfig("Find Regions...", state="normal")
                self.image_menu.entryconfig("Ruler", state="normal")
                self.image_menu.entryconfig("Show H Histogram", state="normal")
                self.image_menu.entryconfig("Show S Histogram", state="normal")
                self.image_menu.entryconfig("Show V Histogram", state="normal")
                self.image_menu.entryconfig("Show R Histogram", state="normal")
                self.image_menu.entryconfig("Show G Histogram", state="normal")
                self.image_menu.entryconfig("Show B Histogram", state="normal")
                self.mask_menu.entryconfig("Background", state="normal")
                self.mask_menu.entryconfig("Matrix", state="normal")
                self.mask_menu.entryconfig("Inclusion", state="normal")
                self.mask_menu.entryconfig("Overwrite", state="normal")
                self.mask_menu.entryconfig("Check Consistency...", state="normal")
                self.mask_menu.entryconfig("Export Statistics...", state="normal")
                self.reset_ruler()
                self.canvas.bind("<ButtonPress-1>", self.start_drag)
                self.canvas.bind("<B1-Motion>", self.do_drag)
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
        if not self.crop_flag.get():
            self.change_status_tool(STATUS_TOOL_NONE)
        else:
            self.change_status_tool(STATUS_TOOL_CROP)

    def pick_color_tool(self):
        """Routine to select the pick color tool."""
        if self.current_mask_color == COLOR_MASK_NONE:
            messagebox.showerror("Error", "No mask class selected.")
            self.pick_color_flag.set(False)
            return
        if not self.pick_color_flag.get():
            self.change_status_tool(STATUS_TOOL_NONE)
        else:
            self.change_status_tool(STATUS_TOOL_PICK_COLOR)
    
    def unpick_color_tool(self):
        """Routine to select the unpick color tool."""
        if not self.unpick_color_flag.get():
            self.change_status_tool(STATUS_TOOL_NONE)
        else:
            self.change_status_tool(STATUS_TOOL_UNPICK_COLOR)

    def pick_region_tool(self):
        """Routine to select the pick region tool."""
        if self.current_mask_color == COLOR_MASK_NONE:
            messagebox.showerror("Error", "No mask class selected.")
            self.pick_region_flag.set(False)
            return
        if not self.pick_region_flag.get():
            self.change_status_tool(STATUS_TOOL_NONE)
        else:
            self.change_status_tool(STATUS_TOOL_PICK_REGION)
            
    def unpick_region_tool(self):
        """Routine to deselect the pick region tool."""
        if not self.unpick_region_flag.get():
            self.change_status_tool(STATUS_TOOL_NONE)
        else:
            self.change_status_tool(STATUS_TOOL_UNPICK_REGION)
            
    def pick_rect_tool(self):
        """Routine to select the pick rectangle region tool."""
        if self.current_mask_color == COLOR_MASK_NONE:
            messagebox.showerror("Error", "No mask class selected.")
            self.pick_rect_flag.set(False)
            return
        if not self.pick_rect_flag.get():
            self.change_status_tool(STATUS_TOOL_NONE)
        else:
            self.change_status_tool(STATUS_TOOL_PICK_RECT)
            
    def unpick_rect_tool(self):
        """Routine to select the unpick rectangle region tool."""
        if not self.unpick_rect_flag.get():
            self.change_status_tool(STATUS_TOOL_NONE)
        else:
            self.change_status_tool(STATUS_TOOL_UNPICK_RECT)

    def ruler_tool(self):
        """Routine to select the ruler tool."""
        if not self.ruler_flag.get():
            self.change_status_tool(STATUS_TOOL_NONE)
        else:
            self.change_status_tool(STATUS_TOOL_RULER)

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
                self.canvas.bind("<ButtonPress-1>", self.start_drag)
                self.canvas.bind("<B1-Motion>", self.do_drag)
                self.status_tool = STATUS_TOOL_NONE
            elif self.status_tool == STATUS_TOOL_PICK_COLOR:
                self.pick_color_flag.set(False)
                self.canvas.config(cursor="arrow")
                self.status_label_message.config(text="")
                self.canvas.unbind("<Button-1>")
                self.canvas.bind("<ButtonPress-1>", self.start_drag)
                self.canvas.bind("<B1-Motion>", self.do_drag)
                self.status_tool = STATUS_TOOL_NONE
            elif self.status_tool == STATUS_TOOL_UNPICK_COLOR:
                self.unpick_color_flag.set(False)
                self.canvas.config(cursor="arrow")
                self.status_label_message.config(text="")
                self.canvas.unbind("<Button-1>")
                self.canvas.bind("<ButtonPress-1>", self.start_drag)
                self.canvas.bind("<B1-Motion>", self.do_drag)
                self.status_tool = STATUS_TOOL_NONE
            elif self.status_tool == STATUS_TOOL_RULER:
                self.ruler_flag.set(False)
                self.canvas.config(cursor="arrow")
                self.status_label_message.config(text="")
                self.canvas.unbind("<Button-1>")
                self.reset_ruler()
                self.canvas.bind("<ButtonPress-1>", self.start_drag)
                self.canvas.bind("<B1-Motion>", self.do_drag)
                self.status_tool = STATUS_TOOL_NONE
            elif self.status_tool == STATUS_TOOL_PICK_REGION:
                self.pick_region_flag.set(False)
                self.canvas.config(cursor="arrow")
                self.status_label_message.config(text="")
                self.canvas.unbind("<Button-1>")
                self.canvas.bind("<ButtonPress-1>", self.start_drag)
                self.canvas.bind("<B1-Motion>", self.do_drag)
                self.status_tool = STATUS_TOOL_NONE
            elif self.status_tool == STATUS_TOOL_UNPICK_REGION:
                self.unpick_region_flag.set(False)
                self.canvas.config(cursor="arrow")
                self.status_label_message.config(text="")
                self.canvas.unbind("<Button-1>")
                self.canvas.bind("<ButtonPress-1>", self.start_drag)
                self.canvas.bind("<B1-Motion>", self.do_drag)
                self.status_tool = STATUS_TOOL_NONE
            elif self.status_tool == STATUS_TOOL_PICK_RECT:
                self.pick_rect_flag.set(False)
                self.canvas.config(cursor="arrow")
                self.status_label_message.config(text="")
                self.canvas.unbind("<Button-1>")
                self.canvas.bind("<ButtonPress-1>", self.start_drag)
                self.canvas.bind("<B1-Motion>", self.do_drag)
                self.status_tool = STATUS_TOOL_NONE
            elif self.status_tool == STATUS_TOOL_UNPICK_RECT:
                self.unpick_rect_flag.set(False)
                self.canvas.config(cursor="arrow")
                self.status_label_message.config(text="")
                self.canvas.unbind("<Button-1>")
                self.canvas.bind("<ButtonPress-1>", self.start_drag)
                self.canvas.bind("<B1-Motion>", self.do_drag)
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
                self.canvas.unbind("<ButtonPress-1>")
                self.canvas.unbind("<B1-Motion>")
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
                self.canvas.unbind("<ButtonPress-1>")
                self.canvas.unbind("<B1-Motion>")
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
                self.canvas.unbind("<ButtonPress-1>")
                self.canvas.unbind("<B1-Motion>")
                # Event for unpicking a color
                self.canvas.bind("<Button-1>", self.on_click)
                self.status_tool = STATUS_TOOL_UNPICK_COLOR
            else:
                messagebox.showerror("Error", "No transition for the status for tools")   
        elif new_status == STATUS_TOOL_RULER:
            # Before chosing a differnt toll, change status to none
            if self.status_tool is not STATUS_TOOL_NONE:
                self.change_status_tool( STATUS_TOOL_NONE)
            if self.status_tool == STATUS_TOOL_NONE:
                self.ruler_flag.set(True)
                self.canvas.config(cursor="cross")
                self.status_label_message.config(text="Click on the image to create a ruler. Esc to cancel.")
                self.canvas.unbind("<ButtonPress-1>")
                self.canvas.unbind("<B1-Motion>")
                # Event for unpicking a color
                self.canvas.bind("<Button-1>", self.on_click_ruler)
                self.status_tool = STATUS_TOOL_RULER
            else:
                messagebox.showerror("Error", "No transition for the status for tools")   
        elif new_status == STATUS_TOOL_PICK_REGION:
            # Before chosing a differnt toll, change status to none
            if self.status_tool is not STATUS_TOOL_NONE:
                self.change_status_tool( STATUS_TOOL_NONE)
            if self.status_tool == STATUS_TOOL_NONE:
                self.pick_region_flag.set(True)
                self.canvas.config(cursor="cross")
                self.status_label_message.config(text="Click on the image to select a region. Esc to cancel.")
                self.canvas.unbind("<ButtonPress-1>")
                self.canvas.unbind("<B1-Motion>")
                # Event for picking a color
                self.canvas.bind("<Button-1>", self.on_click)
                self.status_tool = STATUS_TOOL_PICK_REGION
            else:
                messagebox.showerror("Error", "No transition for the status for tools") 
        elif new_status == STATUS_TOOL_UNPICK_REGION:
            # Before chosing a differnt toll, change status to none
            if self.status_tool is not STATUS_TOOL_NONE:
                self.change_status_tool( STATUS_TOOL_NONE)
            if self.status_tool == STATUS_TOOL_NONE:
                self.unpick_region_flag.set(True)
                self.canvas.config(cursor="cross")
                self.status_label_message.config(text="Click on the mask to deselect a region. Esc to cancel.")
                self.canvas.unbind("<ButtonPress-1>")
                self.canvas.unbind("<B1-Motion>")
                # Event for picking a color
                self.canvas.bind("<Button-1>", self.on_click)
                self.status_tool = STATUS_TOOL_UNPICK_REGION
            else:
                messagebox.showerror("Error", "No transition for the status for tools")  
        elif new_status == STATUS_TOOL_PICK_RECT:
            # Before chosing a differnt toll, change status to none
            if self.status_tool is not STATUS_TOOL_NONE:
                self.change_status_tool( STATUS_TOOL_NONE)
            if self.status_tool == STATUS_TOOL_NONE:
                self.pick_rect_flag.set(True)
                self.canvas.config(cursor="cross")
                self.status_label_message.config(text="Drag the rectangular region to select. Esc to cancel.")
                self.canvas.unbind("<ButtonPress-1>")
                self.canvas.unbind("<B1-Motion>")
                # Events for selectiong a rectangular region
                self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
                self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
                self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
                self.status_tool = STATUS_TOOL_PICK_RECT
            else:
                messagebox.showerror("Error", "No transition for the status for tools")
        elif new_status == STATUS_TOOL_UNPICK_RECT:
            # Before chosing a differnt toll, change status to none
            if self.status_tool is not STATUS_TOOL_NONE:
                self.change_status_tool( STATUS_TOOL_NONE)
            if self.status_tool == STATUS_TOOL_NONE:
                self.unpick_rect_flag.set(True)
                self.canvas.config(cursor="cross")
                self.status_label_message.config(text="Drag the rectangular region to deselect. Esc to cancel.")
                self.canvas.unbind("<ButtonPress-1>")
                self.canvas.unbind("<B1-Motion>")
                # Events for selectiong a rectangular region
                self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
                self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
                self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
                self.status_tool = STATUS_TOOL_UNPICK_RECT
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
                self.canvas.config(cursor="watch")
                self.project.saved = True
                self.project.save(file_path)
                self.file_menu.entryconfig("Save Project", state="disabled")
                self.change_status_tool(STATUS_TOOL_NONE)
                self.canvas.config(cursor="arrow")
        else:
            self.canvas.config(cursor="watch")
            self.project.saved = True
            self.project.save(self.project.project_file)
            self.file_menu.entryconfig("Save Project", state="disabled")
            self.change_status_tool(STATUS_TOOL_NONE)
            self.canvas.config(cursor="arrow")
        self.file_menu.entryconfig("Save Project", state="disabled")

    def save_as_project(self):
        """Routine to save as the project."""
        file_path = filedialog.asksaveasfilename(defaultextension=".prj",
                                            filetypes=[("Project files", "*.prj"), ("All files", "*.*")])
        if file_path:
            self.canvas.config(cursor="watch")
            self.project.saved = True
            self.project.save(file_path)
            self.file_menu.entryconfig("Save Project", state="disabled")
            self.change_status_tool(STATUS_TOOL_NONE)
            self.canvas.config(cursor="arrow")
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
            self.canvas.config(cursor="watch")
            self.project = ProjectData(project_path = file_path, clear_stack_flag = True, console_debug = CONSOLE_DEBUG)
            if self.project.rgb is not None:
                self.change_status(STATUS_IMAGE_LOADED)
                self.zoom_level = 1.0
                self.change_status_display(STATUS_DISPLAY_IMAGE)
                self.display_image()
                self.update_image_size_status(self.project.size)
            self.canvas.config(cursor="arrow")

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
        if self.current_mask_color == COLOR_MASK_NONE:
            messagebox.showerror("Error", "No mask class selected.")
            return
        dialog = PickColorDialog(self.root, self.pick_color_dialog_receive_values, "Pick Color Dialogue", self.pick_color_params)
        dialog.grab_set() # Make the dialog modal
        self.root.wait_window(dialog) # Wait until the dialog is closed

    def open_pick_region_dialog(self):
        """Routine to open the dialogue to set the parameters for the tool pick region."""
        if self.current_mask_color == COLOR_MASK_NONE:
            messagebox.showerror("Error", "No mask class selected.")
            return
        dialog = PickColorDialog(self.root, self.pick_region_dialog_receive_values, "Pick region Dialogue", self.pick_region_params)
        dialog.grab_set() # Make the dialog modal
        self.root.wait_window(dialog) # Wait until the dialog is closed
        
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

    def pick_region_dialog_receive_values(self, values, mode):
        """Routine to process the values from the dialogue to set the parameters for the tool pick region."""
        if values is not None:
            if mode == 'HSV':
                self.pick_region_params['mode'] = 'HSV'
                self.pick_region_params['H'] = values['H']
                self.pick_region_params['S'] = values['S']
                self.pick_region_params['V'] = values['V']
            else:
                self.pick_region_params['mode'] = 'RGB'
                self.pick_region_params['R'] = values['R']
                self.pick_region_params['G'] = values['G']
                self.pick_region_params['B'] = values['B']
                
    def reset_ruler(self, event=None):
        """Routine to reset the ruler.

        Parameters
        ----------
        event : Tkinter event
            Event to process. Default is None.
        """
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

    def on_click_ruler(self, event):
        """Routine to set end of the ruler.

        Parameters
        ----------
        event : Tkinter event
            Event to process.
        """
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
            distance_px = math.hypot(int((end_point[0] - self.start_point[0]) / self.zoom_level), int((end_point[1] - self.start_point[1]) / self.zoom_level))
            distance = distance_px * self.project.conversion_factors.get(self.project.unit, 1.0)
            self.temp_text = self.canvas.create_text((self.start_point[0] + end_point[0]) // 2,
                                                     (self.start_point[1] + end_point[1]) // 2,
                                                     text=f"{distance:.2f} {self.project.unit}", fill="black")
            self.start_point = None
    
    def on_drag_ruler(self, event):
        """Routine to draw the ruler when moving the cursor.

        Parameters
        ----------
        event : Tkinter event
            Event to process.
        """
        if self.start_point:
            if self.temp_line:
                self.canvas.delete(self.temp_line)
            if self.temp_text:
                self.canvas.delete(self.temp_text)
            self.temp_line = self.canvas.create_line(self.start_point[0], self.start_point[1],
                                                     event.x, event.y, fill="gray", dash=(4, 2))
            distance_px = math.hypot(event.x - self.start_point[0], event.y - self.start_point[1])
            distance = distance_px * self.project.conversion_factors.get(self.project.unit, 1.0)
            self.temp_text = self.canvas.create_text((self.start_point[0] + event.x) // 2,
                                                     (self.start_point[1] + event.y) // 2,
                                                     text=f"{distance:.2f} {self.project.unit}", fill="gray")
            
    def open_set_ruler_dialog(self):
        """Routine to open the dialog to calibrate the ruler."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Calibrate Ruler")

        tk.Label(dialog, text="Distance in pixels:").grid(row=0, column=0, padx=5, pady=5)
        pixel_entry = tk.Entry(dialog)
        pixel_entry.grid(row=0, column=1, padx=5, pady=5)
        pixel_entry.insert(0, str(self.project.conversion_factors["px"]))

        tk.Label(dialog, text="Real-world distance:").grid(row=1, column=0, padx=5, pady=5)
        real_entry = tk.Entry(dialog)
        real_entry.grid(row=1, column=1, padx=5, pady=5)
        if self.project.unit != "px":
            unit_var = tk.StringVar(value=self.project.unit)
        else:
            unit_var = tk.StringVar(value="cm")
        if self.project.conversion_factors[unit_var.get()] is not None:
            real_entry.insert(0, str(self.project.conversion_factors[unit_var.get()]))
        else:
            real_entry.insert(0, str(0.0))

        tk.Label(dialog, text="Unit:").grid(row=1, column=2, padx=5, pady=5)
       
        unit_menu = ttk.Combobox(dialog, textvariable=unit_var, values=["cm", "mm", "in"], state="readonly", width=5)
        unit_menu.grid(row=1, column=3, padx=5, pady=5)

        tk.Label(dialog, text="Display unit:").grid(row=2, column=0, padx=5, pady=5)
        display_unit_var = tk.StringVar(value=self.project.unit)
        for i, u in enumerate(["px", "cm", "mm", "in"]):
            tk.Radiobutton(dialog, text=u, variable=display_unit_var, value=u).grid(row=2, column=1+i, padx=2, pady=5)

        def apply_calibration():
            """Routine to apply choices to ruler."""
            try:
                px = float(pixel_entry.get())
                real = float(real_entry.get())
                if px <= 0:
                    raise ValueError("px value must be non-negative")
                if real <= 0:
                    raise ValueError("Unit value must be non-negative")
                unit = unit_var.get()
                factor = real / px
                self.project.conversion_factors["px"] = 1.0
                self.project.conversion_factors["cm"] = factor if unit == "cm" else factor / 10 if unit == "mm" else factor * 2.54
                self.project.conversion_factors["mm"] = self.project.conversion_factors["cm"] * 10
                self.project.conversion_factors["in"] = self.project.conversion_factors["cm"] / 2.54
                self.project.unit = display_unit_var.get()
                self.project.saved = False
                self.file_menu.entryconfig("Save Project", state="normal")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers.")

        tk.Button(dialog, text="Apply", command=apply_calibration).grid(row=3, column=0, columnspan=4, pady=10)

    def open_find_regions_dialog(self):
        """Routine to open the dialogue to automatically find regions."""
        if self.current_mask_color == COLOR_MASK_NONE:
            messagebox.showerror("Error", "No mask class selected.")
            return
        self.find_regions_params["units"] = ["px"]
        if self.project.conversion_factors["cm"] is not None:
            self.find_regions_params["units"].append("cm²")
        if self.project.conversion_factors["mm"] is not None:
            self.find_regions_params["units"].append("mm²")
        if self.project.conversion_factors["in"] is not None:
            self.find_regions_params["units"].append("in²")
            
        dialog = FindRegionsDialog(self.root, self.find_regions_dialog_receive_values, self.project, self.find_regions_params, overwrite_flag=self.overwrite_flag.get())
        dialog.grab_set() # Make the dialog modal
        self.root.wait_window(dialog) # Wait until the dialog is closed

    def find_regions_dialog_receive_values(self, values, mode):
        """Routine to process the values from the dialogue to automatically find regions."""
        if values is not None:
            if mode == 'HSV':
                self.find_regions_params['mode'] = 'HSV'
                self.find_regions_params['min H'] = values['min H']
                self.find_regions_params['min S'] = values['min S']
                self.find_regions_params['min V'] = values['min V']
                self.find_regions_params['max H'] = values['max H']
                self.find_regions_params['max S'] = values['max S']
                self.find_regions_params['max V'] = values['max V']
                range_color = [[values['min H'], values['min S'], values['min V']],
                               [values['max H'], values['max S'], values['max V']]]
            else:
                self.find_regions_params['mode'] = 'RGB'
                self.find_regions_params['min R'] = values['min R']
                self.find_regions_params['min G'] = values['min G']
                self.find_regions_params['min B'] = values['min B']
                self.find_regions_params['max R'] = values['max R']
                self.find_regions_params['max G'] = values['max G']
                self.find_regions_params['max B'] = values['max B']
                range_color = [[values['min R'], values['min G'], values['min B']],
                               [values['max R'], values['max G'], values['max B']]]
                
            self.find_regions_params['edge_threshold1'] = values['edge_min']
            self.find_regions_params['edge_threshold2'] = values['edge_max']
            self.find_regions_params['area_min'] = values['area_min']
            self.find_regions_params['area_max'] = values['area_max']
            self.find_regions_params['area_unit'] = values['area_unit']
            self.find_regions_params['ratio_threshold'] = values['ratio_threshold']

            if self.find_regions_params['area_unit'] == 'px':
                self.find_regions_params['area_min_px'] = self.find_regions_params['area_min']
                self.find_regions_params['area_max_px'] = self.find_regions_params['area_max']
            elif self.find_regions_params['area_unit'] == "cm²":
                if self.project.conversion_factors["cm"] is not None:
                    self.find_regions_params['area_min_px'] = self.find_regions_params['area_min'] * self.project.conversion_factors["cm"]*self.project.conversion_factors["cm"]
                    self.find_regions_params['area_max_px'] = self.find_regions_params['area_max'] * self.project.conversion_factors["cm"]*self.project.conversion_factors["cm"]
                else:
                    messagebox.showerror("Conversion value not set", f"The conversion value for {self.find_regions_params['area_unit']} is not set.")
                    return
            elif self.find_regions_params['area_unit'] == "mm²":
                if self.project.conversion_factors["mm"] is not None:
                    self.find_regions_params['area_min_px'] = self.find_regions_params['area_min'] * self.project.conversion_factors["mm"]*self.project.conversion_factors["mm"]
                    self.find_regions_params['area_max_px'] = self.find_regions_params['area_max'] * self.project.conversion_factors["mm"]*self.project.conversion_factors["mm"]
                else:
                    messagebox.showerror("Conversion value not set", f"The conversion value for {self.find_regions_params['area_unit']} is not set.")
                    return
            elif self.find_regions_params['area_unit'] == "in²":
                if self.project.conversion_factors["in"] is not None:
                    self.find_regions_params['area_min_px'] = self.find_regions_params['area_min'] * self.project.conversion_factors["in"]*self.project.conversion_factors["in"]
                    self.find_regions_params['area_max_px'] = self.find_regions_params['area_max'] * self.project.conversion_factors["in"]*self.project.conversion_factors["in"]
                else:
                    messagebox.showerror("Conversion value not set", f"The conversion value for {self.find_regions_params['area_unit']} is not set.")
                    return
                
            self.canvas.config(cursor="watch")
            self.project.find_regions(self.current_mask_color, range_color, self.find_regions_params['mode'], self.find_regions_params['edge_threshold1'],
                        self.find_regions_params['edge_threshold2'], self.find_regions_params['area_min_px'], self.find_regions_params['area_max_px'],
                        self.find_regions_params['ratio_threshold'] , contour_retrieval=cv2.RETR_TREE,
                        approximation=cv2.CHAIN_APPROX_SIMPLE, overwrite = self.overwrite_flag.get())
            self.canvas.config(cursor="arrow")
            self.update_undo_redo()
            self.file_menu.entryconfig("Save Project", state="normal")
            self.display_image()

    def show_histogram(self, channel_index, mode='RGB'):
        """Routine to show the histogram of one of the channels.

        Parameters
        ----------
        channel_index : integer
            Channel to display.
        mode : str, optional
            Mode to use between HSV and RGB, by default 'RGB'
        """
        if self.status == STATUS_IMAGE_LOADED:
            if mode=='RGB':
                img_tmp = np.array(self.project.rgb)
                if channel_index == 0:
                    color = "red"
                    title = "Red Histogram"
                elif channel_index == 1:
                    color = "green"
                    title = "Green Histogram"
                elif channel_index == 2:
                    color = "blue"
                    title = "Blue Histogram"
                else:
                    messagebox.showerror("Error", "Channel not recognised.")
                    return
                channel = img_tmp[:, :, channel_index]
                plt.figure()
                plt.hist(channel.ravel(), bins=256, range=(0, 256), color=color)
                plt.title(title)
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.show()
            elif mode=='HSV':
                if channel_index == 0:
                    color = "purple"
                    title = "Hue Histogram"
                    max_value = 180
                elif channel_index == 1:
                    color = "gray"
                    title = "Saturation Histogram"
                    max_value = 256
                elif channel_index == 2:
                    color = "yellow"
                    title = "Value Histogram"
                    max_value = 256
                else:
                    messagebox.showerror("Error", "Channel not recognised.")
                    return
                channel = self.project.hsv[:, :, channel_index]
                plt.figure()
                plt.hist(channel.ravel(), bins=max_value, range=(0, max_value), color=color)
                plt.title(title)
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.show()
            else:
                messagebox.showerror("Error", "Mode not recognised.")
                return
    def show_h_histogram(self):
        """Routine to show the H histogram"""
        self.show_histogram(0, "HSV")

    def show_s_histogram(self):
        """Routine to show the S histogram"""
        self.show_histogram(1, "HSV")

    def show_v_histogram(self):
        """Routine to show the V histogram"""
        self.show_histogram(2, "HSV")
        
    def show_r_histogram(self):
        """Routine to show the R histogram"""
        self.show_histogram(0, "RGB")

    def show_g_histogram(self):
        """Routine to show the G histogram"""
        self.show_histogram(1, "RGB")

    def show_b_histogram(self):
        """Routine to show the B histogram"""
        self.show_histogram(2, "RGB")
        
    def open_color_range_dialog(self):
        """Routine to open the dialogue to select color range."""
        if self.current_mask_color == COLOR_MASK_NONE:
            messagebox.showerror("Error", "No mask class selected.")
            return
        dialog = ColorRangeDialog(self.root, self.color_range_dialog_receive_values, self.color_range_params)
        dialog.grab_set() # Make the dialog modal
        self.root.wait_window(dialog) # Wait until the dialog is closed

    def color_range_dialog_receive_values(self, values, mode):
        """Routine to process the values from the dialogue to select color range."""
        if values is not None:
            if mode == 'HSV':
                self.color_range_params['mode'] = 'HSV'
                self.color_range_params['min H'] = values['min H']
                self.color_range_params['min S'] = values['min S']
                self.color_range_params['min V'] = values['min V']
                self.color_range_params['max H'] = values['max H']
                self.color_range_params['max S'] = values['max S']
                self.color_range_params['max V'] = values['max V']
                range_color = [[values['min H'], values['min S'], values['min V']],
                               [values['max H'], values['max S'], values['max V']]]
            else:
                self.color_range_params['mode'] = 'RGB'
                self.color_range_params['min R'] = values['min R']
                self.color_range_params['min G'] = values['min G']
                self.color_range_params['min B'] = values['min B']
                self.color_range_params['max R'] = values['max R']
                self.color_range_params['max G'] = values['max G']
                self.color_range_params['max B'] = values['max B']
                range_color = [[values['min R'], values['min G'], values['min B']],
                               [values['max R'], values['max G'], values['max B']]]
                
            self.canvas.config(cursor="watch")
            self.project.color_range(self.current_mask_color, range_color, self.color_range_params['mode'], overwrite = self.overwrite_flag.get())
            self.canvas.config(cursor="arrow")
            self.update_undo_redo()
            self.file_menu.entryconfig("Save Project", state="normal")
            self.display_image()
            
    def start_drag(self, event):
        """Routine to start dragging the image with the mouse.

        Parameters
        ----------
        event : event object
            Position of the mouse.
        """
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def do_drag(self, event):
        """Routine to drag the image with the mouse.

        Parameters
        ----------
        event : event object
            Position of the mouse.
        """
        dx = self.drag_start_x - event.x
        dy = self.drag_start_y - event.y
        self.canvas.xview_scroll(int(dx), "units")
        self.canvas.yview_scroll(int(dy), "units")
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
    def show_splash(self,root, image_path, duration=3000):
        """Routine to show the splash window.

        Parameters
        ----------
        root : TK object
            TK object owning the splash window.
        image_path : string
            Path to the image to show.
        duration : int, optional
            Duration of the splash screen, by default 3000
        """
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
        tk.Label(text_frame, text="Authors: Lucy Standish and Stefano Giani.", bg="white", font=("Arial", 12)).pack(pady=(10, 0))
        tk.Label(text_frame, text="License: CC-BY-NC-4.0", bg="white", font=("Arial", 10)).pack()

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
    
    def show_about_window(self, root, image_path):
        """Routine to show the about window.

        Parameters
        ----------
        root : TK object
            TK object owning the splash window.
        image_path : string
            Path to the image to show.
        """
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
        tk.Label(text_frame, text="Authors: Stefano Giani and Lucy Standish", bg="white", font=("Arial", 12)).pack(pady=(10, 0))
        tk.Label(text_frame, text="License: CC-BY-NC-4.0", bg="white", font=("Arial", 10)).pack()

        ttk.Button(about, text="Close", command=about.destroy).pack(pady=10)

        about.update_idletasks()
        width = about.winfo_width()
        height = about.winfo_height()
        x = (about.winfo_screenwidth() // 2) - (width // 2)
        y = (about.winfo_screenheight() // 2) - (height // 2)
        about.geometry(f"{width}x{height}+{x}+{y}")
        
    def check_consistency(self):
        """Routine to check if all the pixels in the mask are assigned.
        """
        if self.project.check_mask_consistency():
            messagebox.showinfo("Mask consistency", "All the pixels in the mask are assigned to classes.")
        else:
            popup = CustomPopupMaskConsistency(self.root)
            if popup.choice == 'Background':
                self.canvas.config(cursor="watch")
                self.project.select_color_mask(COLOR_MASK_NONE, COLOR_MASK_NONE, COLOR_MASK_BACKGROUND, overwrite = False)
                self.canvas.config(cursor="arrow")
                self.update_undo_redo()
                self.file_menu.entryconfig("Save Project", state="normal")
                self.display_image()
            elif popup.choice == 'Matrix':
                self.canvas.config(cursor="watch")
                self.project.select_color_mask(COLOR_MASK_NONE, COLOR_MASK_NONE, COLOR_MASK_MATRIX, overwrite = False)
                self.canvas.config(cursor="arrow")
                self.update_undo_redo()
                self.file_menu.entryconfig("Save Project", state="normal")
                self.display_image()
            elif popup.choice == 'Inclusions':
                self.canvas.config(cursor="watch")
                self.project.select_color_mask(COLOR_MASK_NONE, COLOR_MASK_NONE, COLOR_MASK_INCLUSION, overwrite = False)
                self.canvas.config(cursor="arrow")
                self.update_undo_redo()
                self.file_menu.entryconfig("Save Project", state="normal")
                self.display_image()
                
    def export_statistics(self):
        """Routine to export statistics of the inclusions ot file.
        """
        if not self.project.check_mask_consistency():
            messagebox.showwarning("Warning", "Not all the pixels in the mask are assigned to classes.")
            
        self.canvas.config(cursor="watch")
        regions = self.project.extract_regions(COLOR_MASK_INCLUSION)
        self.canvas.config(cursor="arrow")
        file_path = filedialog.asksaveasfilename(defaultextension=".h5",
                                                filetypes=[("H5", "*.h5"), ("Pickle", "*.pkl"), ("All files", "*.*")])
        extension = os.path.splitext(file_path)[1]
        if extension == '.h5':
            self.canvas.config(cursor="watch")
            self.project.save_regions_stats_h5(regions, file_path)
            self.canvas.config(cursor="arrow")
        elif extension == '.pkl':
            self.canvas.config(cursor="watch")
            self.project.save_regions_stats_pickle(regions, file_path)
            self.canvas.config(cursor="arrow")
        else:
            messagebox.showerror("Error", "Extension not recognised")
    
    def create_tiles(self):
        if not self.project.check_mask_consistency():
            messagebox.showwarning("Warning", "Not all the pixels in the mask are assigned to classes.")
        
        dialog = tk.Toplevel(root)
        dialog.title("Tile Parameters Input")
        
        def choose_output_dir():
            selected_dir = filedialog.askdirectory()
            if selected_dir:
                entry_output_dir.delete(0, tk.END)
                entry_output_dir.insert(0, selected_dir)

        def submit():
            try:
                tile_width = int(entry_tile_width.get())
                tile_height = int(entry_tile_height.get())
                overlap_x = int(entry_overlap_x.get())
                overlap_y = int(entry_overlap_y.get())
                tiles_name = entry_tiles_name.get()
                output_dir = entry_output_dir.get()

                # Validation
                if tile_width > self.project.size[0]:
                    raise ValueError("Tile width exceeds image width.")
                if tile_height > self.project.size[1]:
                    raise ValueError("Tile height exceeds image height.")
                if overlap_x >= tile_width:
                    raise ValueError("Horizontal overlap must be less than tile width.")
                if overlap_y >= tile_height:
                    raise ValueError("Vertical overlap must be less than tile height.")
                if not tiles_name:
                    raise ValueError("Tiles name cannot be empty.")
                if not output_dir:
                    raise ValueError("Output directory cannot be empty.")

                self.canvas.config(cursor="watch")
                self.project.create_tiles(tile_width, tile_height, overlap_x, overlap_y, tiles_name, output_dir)
                self.canvas.config(cursor="arrow")
                
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Input Error", str(e))

        # Create and place labels and entry widgets
        tk.Label(dialog, text="Tile Width:").grid(row=0, column=0)
        entry_tile_width = tk.Entry(dialog)
        entry_tile_width.grid(row=0, column=1)

        tk.Label(dialog, text="Tile Height:").grid(row=1, column=0)
        entry_tile_height = tk.Entry(dialog)
        entry_tile_height.grid(row=1, column=1)

        tk.Label(dialog, text="Horizontal Overlap:").grid(row=2, column=0)
        entry_overlap_x = tk.Entry(dialog)
        entry_overlap_x.grid(row=2, column=1)

        tk.Label(dialog, text="Vertical Overlap:").grid(row=3, column=0)
        entry_overlap_y = tk.Entry(dialog)
        entry_overlap_y.grid(row=3, column=1)

        tk.Label(dialog, text="Tiles Name:").grid(row=4, column=0)
        entry_tiles_name = tk.Entry(dialog)
        entry_tiles_name.grid(row=4, column=1)

        # Output Directory
        tk.Label(dialog, text="Output Directory:").grid(row=5, column=0)
        entry_output_dir = tk.Entry(dialog)
        entry_output_dir.grid(row=5, column=1)
        tk.Button(dialog, text="Browse", command=choose_output_dir).grid(row=5, column=2)

        # Submit button
        tk.Button(dialog, text="Submit", command=submit).grid(row=6, column=0, columnspan=2)

# Run the app
root = tk.Tk()
app = PetroSeg(root)
root.mainloop()
