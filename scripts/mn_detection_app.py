import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

from mn_segmentation.lib.Application import Application

categories = ["micronuclei"]

@st.cache_resource
def load_model():
    model = Application(weight="model/RCNN.pt")
    return model

model = load_model()

def make_prediction(img): 
    prediction = model._predict(img) ## Dictionary with keys "boxes", "labels", "scores".
    # prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction): ## Adds Bounding Boxes around original Image.
    img_tensor = torch.tensor(img) ## Transpose
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction, width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0) ### (3,W,H) -> (W,H,3), Channel first to channel last.
    return img_with_bboxes_np

## Dashboard
st.title("Micronuclei Detector :tea: :coffee:")
upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload)

    image = pil_to_tensor(img)
    prediction = make_prediction(image) ## Dictionary
    pred_boxes, pred_masks = model._post_process(prediction)
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2,0,1), pred_boxes) ## (W,H,3) -> (3,W,H)

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


