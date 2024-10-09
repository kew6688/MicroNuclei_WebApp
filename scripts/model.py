import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import streamlit as st

from mn_segmentation.lib.Application import Application

@st.cache_resource
def load_model(weight):
    model = Application(weight)
    return model

class Model:
    def __init__(self, model_path):
        self.app = load_model(weight=model_path)
        self.categories = ["micronuclei"]

    def make_prediction(self, img): 
        '''
        Call ML model to make a prediction
        '''
        prediction = self.app._predict(img) ## Dictionary with keys "boxes", "labels", "scores".
        # prediction["labels"] = [categories[label] for label in prediction["labels"]]
        return prediction

    def create_image_with_bboxes(self, img, prediction): 
        '''
        Adds Bounding Boxes around original Image.
        '''
        img_tensor = torch.tensor(img) ## Transpose
        img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], width=2)
        img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0) ### (3,W,H) -> (W,H,3), Channel first to channel last.
        return img_with_bboxes_np

    def process_image(self, img):
        '''
        Take input image from view, crop the image into fixed size, predict and create image stitched back.
        '''
        image = pil_to_tensor(img)
        prediction = self.make_prediction(image) ## Dictionary
        img_with_bbox = self.create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction) ## (W,H,3) -> (3,W,H)
        return prediction, img_with_bbox