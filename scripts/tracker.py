import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from ipywidgets import Button, HBox, VBox, Output
import os
from PIL import Image

class Tracker:
    '''
    Tracker class handles track functions and trajectory displays
    '''
    def __init__(self):
        pass

    def display(self, frames, points):
        '''
        display tracking points over frames, one point per frame

        Parameters:
            frames: list of frames, each frame is a numpy array of Image.open(path)
            points: list of coordinates, [ [x,y] , ...]
        '''

        # Function to interpolate a smooth curve between all points from frame 0 to current frame
        def interpolate_curve(points_up_to_current):
            # Separate the list of points into x and y coordinates
            x = [p[0] for p in points_up_to_current]
            y = [p[1] for p in points_up_to_current]
            
            # Use spline interpolation to get a smooth curve
            tck, u = splprep([x, y], s=0)  # s=0 for smooth curve passing through all points
            u_fine = np.linspace(0, 1, 100)  # Parameter for generating more points along the curve
            x_fine, y_fine = splev(u_fine, tck)
            
            return x_fine, y_fine

        # Function to display the current frame, point, and a smooth curve up to the current frame
        def show_frame():
            global current_frame_index
            with output:
                output.clear_output(wait=True)  # Clear the previous frame
                plt.imshow(frames[current_frame_index], cmap='gray')

                # Plot the current point
                current_point = points[current_frame_index]
                plt.scatter(current_point[0], current_point[1], c='red', s=10)  # Adjust color and size of the point

                # Draw a line to the previous point if not on the first frame
                if current_frame_index > 0:
                    previous_point = points[current_frame_index - 1]
                    plt.plot([previous_point[0], current_point[0]], [previous_point[1], current_point[1]], 'blue')  # Line in blue

                # Draw a line through all the previous point if not on the first three frames
                if current_frame_index > 2:
                    # Interpolate and draw a smooth curve through all points up to the current frame
                    points_up_to_current = points[:current_frame_index + 1]  # Get all points from frame 0 to current frame
                    x_fine, y_fine = interpolate_curve(points_up_to_current)
                    plt.plot(x_fine, y_fine, 'red')  # Draw the smooth curve in green

                plt.title(f"Frame {folder[current_frame_index][:-4]} with Point {current_point}")
                plt.axis('off')
                plt.show()

        # Callback for 'Previous' button
        def on_previous_clicked(b):
            global current_frame_index
            if current_frame_index > 0:
                current_frame_index -= 1
                show_frame()

        # Callback for 'Next' button
        def on_next_clicked(b):
            global current_frame_index
            if current_frame_index < len(frames) - 1:
                current_frame_index += 1
                show_frame()

        # Example list of image frames 
        frames = frames
            
        # Example list of points (one point per frame)
        points = [(755, 467), (772, 448), (750, 435), (690, 405), (650, 375), (630, 371), (600, 373), (540, 355), (485, 305), (433, 266)]  # example points

        current_frame_index = 0  # Global variable to track the current frame index

        # Create an output widget to display the plot
        output = Output()

        # Create 'Previous' and 'Next' buttons
        previous_button = Button(description="Previous")
        next_button = Button(description="Next")

        # Attach the callback functions to the buttons
        previous_button.on_click(on_previous_clicked)
        next_button.on_click(on_next_clicked)

        # Layout buttons and output in the notebook
        ui = HBox([previous_button, next_button])
        display(VBox([ui, output]))

        # Show the first frame initially
        show_frame()
