import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

class View:
    '''
    View handles UI and interact functions. 
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
        upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

        if upload:
            img = Image.open(upload)
            
            # Define the canvas properties
            canvas_width = 600
            canvas_height = 400
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Convert the uploaded image to a drawable canvas background
            img_width, img_height = img.size
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

            if st.button('Process Image'):
                start = time.time()
                prediction, img_with_bbox = self.model.process_image_seg_mn(img)
                print(f"segment mn takes {time.time() - start}")

                start = time.time()
                masks = self.model.process_image_seg_nuc(img)
                print(f"segment nuc takes {time.time() - start}")

                fig = plt.figure(figsize=(12,12))
                ax = fig.add_subplot(111)
                plt.imshow(img_with_bbox)
                plt.xticks([],[])
                plt.yticks([],[])
                ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

                show_anns(masks)

                st.pyplot(fig, use_container_width=True)

                st.header("Predictions")
                st.write(prediction)
            
