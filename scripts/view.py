import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
        ## Dashboard
        st.title("Micronuclei Detector :tea: :coffee:")
        upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

        if upload:
            img = Image.open(upload)

            prediction, img_with_bbox = self.model.process_image(img)

            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot(111)
            plt.imshow(img_with_bbox)
            plt.xticks([],[])
            plt.yticks([],[])
            ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

            st.pyplot(fig, use_container_width=True)

            del prediction["boxes"]
            st.header("Predicted Probabilities")
            st.write(prediction)


