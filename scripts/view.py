import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from scipy.interpolate import splprep, splev

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        if ann['area'] > 50000: continue
        m = ann['segmentation']
        color_mask = np.concatenate([np.array([255,255,255]), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf,format='PNG')
    buf.seek(0)
    img = Image.open(buf)
    print(fig.canvas.get_width_height())
    return img

def set_process():
    st.session_state.process = True

@st.fragment
def show_predict_outputs():
    # if st.session_state.image_uploaded:
    #     st.rerun()

    if not st.session_state.clicked:
        with st.container(height=510): ### ADD CONTAINER ###
            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot(111)
            plt.imshow(st.session_state.img_with_bbox)
            plt.xticks([],[])
            plt.yticks([],[])
            ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

            # show_anns(st.session_state.masks)

            st.pyplot(fig, use_container_width=True)
            # time.sleep(2)

            st.header("Predictions")
            st.write(st.session_state.prediction)
    else:
        with st.container(height=510): ### ADD CONTAINER ###
            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot(111)
            plt.imshow(st.session_state.img_with_bbox)
            plt.xticks([],[])
            plt.yticks([],[])
            ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

            show_anns(st.session_state.masks)

            st.pyplot(fig, use_container_width=True)
            # time.sleep(2)

            st.header("Predictions")
            st.write(st.session_state.prediction)
        
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

class DetectView:
    '''
    View class handles UI and interact functions. 
    '''
    def __init__(self, model):
        self.model = model
    
    def run(self):
        '''
        Render the UI and start app.
        '''
        print("======================= UI start =======================")
        ## Dashboard
        st.title("Micronuclei Detector :tea: :coffee:")

        # Once we have the dependencies, add a selector for the app mode on the sidebar.
        st.sidebar.title("What to do")
        app_mode = st.sidebar.selectbox("Choose the app mode",
            ["Show instructions", "Run the app", "Show the source code"])
        if app_mode == "Show instructions":
            st.sidebar.success('To continue select "Run the app".')
        elif app_mode == "Show the source code":
            pass
        elif app_mode == "Run the app":
            self.run_the_app()
            

    def run_the_app(self):
        resolution = self.resolution_ui()
        conf,iou = self.object_detector_ui()
        if 'image_uploaded' not in st.session_state:
            st.session_state.image_uploaded = False
            st.session_state.last_uploaded_image = None

        upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg", "tif"])
        
        if 'clicked' not in st.session_state:
            st.session_state.clicked = False

        def click_button():
            st.session_state.clicked = not st.session_state.clicked
        st.button('Display nuclei segmentation', on_click=click_button)

        if upload:
            if st.session_state.last_uploaded_image != upload:
                st.session_state.image_uploaded = True
                st.session_state.last_uploaded_image = upload
            else:
                st.session_state.image_uploaded = False

            img = Image.open(upload)

            if st.session_state.image_uploaded:
                start = time.time()
                prediction, img_with_bbox = self.model.process_image_seg_mn(img)
                print(f"segment mn takes {time.time() - start}")

                start = time.time()
                masks = self.model.process_image_seg_nuc(img)
                print(f"segment nuc takes {time.time() - start}")

                st.session_state.prediction = prediction
                st.session_state.img_with_bbox = img_with_bbox
                st.session_state.masks = masks

            st.session_state.img = show_predict_outputs()
            

            # with st.container(height=1000): ### ADD CONTAINER ###
            # if st.session_state.image_uploaded:
            # Define the canvas properties
            canvas_width = 800
            canvas_height = 600
            # st.image(st.session_state.img, caption="Predicted Image", use_column_width=True)

            st.header("Predictions")
            st.download_button('Download output', json.dumps(st.session_state.prediction)) # TODO: use proper json format
            st.write(st.session_state.prediction)

    def resolution_ui(self):
        st.sidebar.markdown("# Image Resolution (um/pixel)")
        # Create an input box in the sidebar
        user_input = st.sidebar.text_input("Enter your image resolution")

        # Display the input in the main page
        st.write(f"You image resolution is: {user_input} um/pixel")

    def object_detector_ui(self):
        st.sidebar.markdown("# Model")
        confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
        overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
        return confidence_threshold, overlap_threshold

class TrackView:
    '''
    View class handles UI and interact functions. 
    '''
    def __init__(self, model):
        self.model = model
    
    def run(self):
        '''
        Render the UI and start app.
        '''
        print("======================= Track UI start =======================")
        ## Dashboard
        st.title("Micronuclei Detector :tea: :coffee:")

        # Once we have the dependencies, add a selector for the app mode on the sidebar.
        st.sidebar.title("What to do")
        app_mode = st.sidebar.selectbox("Choose the app mode",
            ["Show instructions", "Run the app", "Show the source code"])
        if app_mode == "Show instructions":
            st.sidebar.success('To continue select "Run the app".')
        elif app_mode == "Show the source code":
            pass
        elif app_mode == "Run the app":
            self.run_the_app()

    def run_the_app(self):
        st.sidebar.markdown("# Frame")

        # main canvas
        if 'image_uploaded' not in st.session_state:
            st.session_state.image_uploaded = False
            st.session_state.last_uploaded_image = None

        upload = st.file_uploader(label="Upload Frames Here:", accept_multiple_files=True, type=["png", "jpg", "jpeg", "tif"])
        # print(upload[0].name)
        upload.sort(key=lambda x: x.name)
        if upload:
            # step 1: load pre points
            points = [(755, 467), (772, 448), (750, 435), (690, 405), (650, 375), (630, 371), (600, 373), (540, 355), (485, 305), (433, 266)]  # example points

            # Step 2: Use a slider to select which image to display
            frame_idx = st.sidebar.slider("Select frame", 0, len(upload) - 1, 0)

            # Step 3: Open the selected image using PIL
            img = Image.open(upload[frame_idx])

            # Step 4: Display the selected image
            # st.image(img, caption=f"Frame {frame_idx + 1}", use_column_width=True)
            with st.container(height=510): ### ADD CONTAINER ###
                fig = plt.figure(figsize=(12,12))
                ax = fig.add_subplot(111)
                frame = np.array(img)
                plt.imshow(frame)

                # Plot the current point
                current_point = points[frame_idx]
                plt.scatter(current_point[0], current_point[1], c='red', s=10)  # Adjust color and size of the point

                # Draw a line to the previous point if not on the first frame
                if frame_idx > 0:
                    previous_point = points[frame_idx - 1]
                    plt.plot([previous_point[0], current_point[0]], [previous_point[1], current_point[1]], 'blue')  # Line in blue

                # Draw a line through all the previous point if not on the first three frames
                if frame_idx > 2:
                    # Interpolate and draw a smooth curve through all points up to the current frame
                    points_up_to_current = points[:frame_idx + 1]  # Get all points from frame 0 to current frame
                    x_fine, y_fine = interpolate_curve(points_up_to_current)
                    plt.plot(x_fine, y_fine, 'red')  # Draw the smooth curve in green

                plt.title(f"Frame {frame_idx} with Point {current_point}")
                plt.axis('off')
                plt.show()
                st.pyplot(fig, use_container_width=True)

            # with st.container(height=1000): ### ADD CONTAINER ###
            # if st.session_state.image_uploaded:
            # Define the canvas properties
            canvas_width = 800
            canvas_height = 600

            st.header("Track")
            # Convert the uploaded image to a drawable canvas background
            img_width, img_height = img.size
            print(img.size)
            scale_factor = canvas_width / img_width
            resized_image = img.resize((canvas_width, int(img_height * scale_factor)))

            # Create the drawing canvas on top of the image
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Color of the drawing (semi-transparent orange)
                stroke_width=3,  # Width of the stroke
                stroke_color="black",  # Color of the stroke
                background_image=resized_image,  # The background image to draw on
                update_streamlit=True,
                width=canvas_width,
                height=canvas_height,
                drawing_mode="point",  # Set to "point" to allow to click points
                key="canvas",
            )

            # Display a message after drawing
            if canvas_result.image_data is not None:
                st.image(canvas_result.image_data, caption="Edited Image")

            # Display point-adding functionality
            points = []
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                for obj in objects:
                    if obj["type"] == "circle":
                        x, y = obj["left"], obj["top"]
                        points.append((x, y))  # Append coordinates to the points list
            # Show the collected points
            st.write("Collected Points:", points)
        else:
            st.write("Upload some images to display them here.")