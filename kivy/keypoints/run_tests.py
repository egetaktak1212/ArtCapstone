import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

import math

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from isolate_face import isolate_face

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_landmark_model.pth")
DB_PATH    = os.path.join(BASE_DIR, "aflw.sqlite")
IMAGES_PATH = os.path.join(BASE_DIR, "image", "cameron.JPG")
IMG_SIZE      = 128
NUM_LANDMARKS = 21
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  load model
class LandmarkResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 42)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

model = LandmarkResNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.eval()
print("Model loaded")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

LANDMARK_NAMES = [
    "l_brow_left", "l_brow_mid", "l_brow_right",
    "r_brow_left", "r_brow_mid", "r_brow_right",
    "l_eye_left", "l_eye_mid", "l_eye_right",
    "r_eye_left", "r_eye_mid", "r_eye_right",
    "l_ear","nose_left", "nose_mid", "nose_right", "r_ear",
    "mouth_left", "mouth_mid", "mouth_right",
    "chin"
]

# predict landmarks
def predict_and_show(img_path, bbox):
    bx, by, bw, bh = bbox

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load: {img_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]

    # crop face
    bx = max(0, bx);  by = max(0, by)
    bw = min(bw, img_w - bx);  bh = min(bh, img_h - by)
    face = img[by:by+bh, bx:bx+bw]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    # Run model
    tensor = transform(face).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()

    results = []
    for i in range(NUM_LANDMARKS):
        rel_x = float(pred[i*2])# [0,1] relative to crop
        rel_y = float(pred[i*2+1])

        # Pixel coords in the full image
        px = int(bx + rel_x * bw)
        py = int(by + rel_y * bh)

    
        results.append({
            "landmark_id":   i + 1,
            "name":          LANDMARK_NAMES[i],
            "pixel_x":       px,
            "pixel_y":       py,
        })


    points_dict = {r["landmark_id"]: (r["pixel_x"], r["pixel_y"]) for r in results}
    points_dict[22] = (math.floor((points_dict[9][0] + points_dict[10][0])/2), math.floor((points_dict[3][1] + points_dict[9][1])/2))

    return points_dict


# run on image
def run_everything(image_path):
    sys.path.insert(0, os.path.dirname(__file__))
    from cnn_training_2 import load_data


    data = load_data(DB_PATH)

    full_path = image_path
    bbox = isolate_face(full_path)

    image_shape = cv2.imread(full_path).shape
    
    if bbox is None:
        bbox = (0,0,image_shape[0], image_shape[1])

    coords = predict_and_show(full_path, bbox)

    return coords, bbox
