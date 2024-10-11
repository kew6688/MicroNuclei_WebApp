import streamlit as st
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


