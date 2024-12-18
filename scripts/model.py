import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import streamlit as st

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from mn_segmentation.lib.Application import Application
import mn_segmentation.lib.cluster as cluster

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


@st.cache_resource
def load_model_mn(weight):
    model = Application(weight)
    return model

@st.cache_resource
def load_model_sam(weight,cfg):
    sam2_checkpoint = weight
    model_cfg = cfg

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    nuc_app = SAM2AutomaticMaskGenerator(sam2)
    # nuc_app = SAM2AutomaticMaskGenerator(
    #     model=sam2,
    #     points_per_side=64,
    #     points_per_batch=128,
    #     pred_iou_thresh=0.7,
    #     stability_score_thresh=0.92,
    #     stability_score_offset=0.7,
    #     min_mask_region_area=25
    # )
    return nuc_app

class Model:
    '''
    Model class handels backend functions and load machine learning models
    '''
    def __init__(self, mn_model_path, nuc_model_path, nuc_model_cfg):
        print("======================= model start =======================")
        self.mn_app = load_model_mn(weight=mn_model_path)
        self.nuc_app = load_model_sam(nuc_model_path,nuc_model_cfg)
        self.categories = ["nuclei","micronuclei"]

    def make_prediction(self, img): 
        '''
        Call ML model to make a prediction
        '''
        prediction = self.mn_app._predict(img) ## Dictionary with keys "boxes", "labels", "scores".
        prediction["labels"] = [self.categories[label] for label in prediction["labels"]]
        return prediction

    def create_image_with_bboxes(self, img, prediction): 
        '''
        Adds Bounding Boxes around original Image.
        '''
        img_tensor = torch.tensor(img) ## Transpose
        prediction = torch.tensor(prediction)
        img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction, colors="red", width=1)
        img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0) ### (3,W,H) -> (W,H,3), Channel first to channel last.
        return img_with_bboxes_np

    def process_image_seg_mn(self, img):
        '''
        Take input image from view, crop the image into fixed size, predict and create image stitched back.
        '''
        print("======================= process image =======================")

        # crop image to model input size 224x224
        wnd_sz = 224
        height, width = np.array(img).shape[:2]

        # sliding a window to process the image
        boxes_lst = []
        output = {"coord":[], "area":[], "bbox":[]}
        for i in range(height // wnd_sz + 1):
            for j in range(width // wnd_sz + 1):

                cur_x, cur_y = wnd_sz * j, wnd_sz * i
                box = (cur_x, cur_y, cur_x + wnd_sz, cur_y + wnd_sz)

                image = pil_to_tensor(img.crop(box))
                prediction = self.make_prediction(image) ## Dictionary
                pred_boxes, pred_masks,_ = self.mn_app._post_process(prediction)
                pred_boxes[:, [0,2]] += cur_x
                pred_boxes[:, [1,3]] += cur_y
                boxes_lst.append(pred_boxes.cpu().numpy())

                area = pred_masks.sum(1).sum(1).sum(1)
                output["bbox"] += pred_boxes.cpu().numpy().tolist()
                output["coord"] += cluster.boxToCenters(pred_boxes).tolist()
                output["area"] += area.cpu().numpy().tolist()
                
        pred_boxes = np.concatenate(boxes_lst)
        print(pred_boxes, pred_boxes.shape)
        img_with_bbox = self.create_image_with_bboxes(np.array(img)[:,:,:3].transpose(2,0,1), pred_boxes) ## (W,H,3) -> (3,W,H)
        return output, img_with_bbox
    
    def process_image_seg_nuc(self,img):
        img = np.array(img.convert("RGB"))
        masks = self.nuc_app.generate(img)
        return masks