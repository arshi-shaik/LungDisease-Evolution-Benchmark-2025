pip install grad-cam

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np

def generate_gradcam(model, img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    target_layers = [model.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=torch.tensor(img_rgb).unsqueeze(0).permute(0,3,1,2))
    visualize = show_cam_on_image(img_rgb, grayscale_cam[0], use_rgb=True)

    cv2.imwrite("results/gradcam_output.png", visualize)
    return visualize
