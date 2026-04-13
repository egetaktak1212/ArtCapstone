import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import sqlite3
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DB_PATH      = "C:/Users/miamc/Downloads/AFLW_folder/AFLW/Face_Detection_on_AFLW_dataset/src/aflw.sqlite"
IMAGES_PATH  = "C:/Users/miamc/Downloads/AFLW_folder/AFLW/Face_Detection_on_AFLW_dataset/src/flickr/"
IMG_SIZE     = 128
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 0.0001
NUM_LANDMARKS = 21

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    # Get bounding boxes
    bbox_query = """
        SELECT faceimages.filepath, facerect.x, facerect.y, facerect.w, facerect.h, faces.face_id
        FROM faceimages, faces, facerect
        WHERE faces.file_id = faceimages.file_id
        AND facerect.face_id = faces.face_id
    """
    bboxes = {}
    for row in c.execute(bbox_query):
        filepath, x, y, w, h, face_id = row
        bboxes[face_id] = (filepath, x, y, w, h)

    # Get landmarks
    landmark_query = """
        SELECT face_id, feature_id, x, y
        FROM featurecoords
        ORDER BY face_id, feature_id
    """
    landmarks = {}
    for row in c.execute(landmark_query):
        face_id, feature_id, x, y = row
        if face_id not in landmarks:
            landmarks[face_id] = {}
        landmarks[face_id][feature_id] = (x, y)

    # Get pose
    poses = {}
    for row in c.execute("SELECT face_id, yaw FROM facepose"):
        poses[row[0]] = row[1]

    conn.close()

    # Only keep faces that have BOTH bbox AND at least some landmarks
    data = []
    for face_id, (filepath, bx, by, bw, bh) in bboxes.items():
        lm = landmarks.get(face_id, {})
        if len(lm) == 0:
            continue  # skip faces with no landmarks at all
        data.append({
            'filepath':  filepath,
            'bbox':      (bx, by, bw, bh),
            'landmarks': lm,
            'yaw':       poses.get(face_id, 0.0)
        })

    frontal  = sum(1 for d in data if abs(d['yaw']) < math.radians(30))
    sideways = sum(1 for d in data if abs(d['yaw']) >= math.radians(30))
    profile  = sum(1 for d in data if abs(d['yaw']) >= math.radians(60))

    return data

# ── 2. DATASET ────────────────────────────────────────────────────────────────
class AFLWDataset(Dataset):
    def __init__(self, data, images_path, img_size=128, transform=None):
        self.data          = data
        self.images_path   = images_path
        self.img_size      = img_size
        self.transform     = transform
        self.num_landmarks = NUM_LANDMARKS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item             = self.data[idx]
        img_path         = self.images_path + item['filepath']
        bx, by, bw, bh   = item['bbox']

        # Load full image
        img = cv2.imread(img_path)
        if img is None:
            img      = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            lm_label = np.zeros(self.num_landmarks * 2, dtype=np.float32)
            mask     = np.zeros(self.num_landmarks * 2, dtype=np.float32)
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(lm_label), torch.tensor(mask)

        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        # ── KEY FIX: Crop the face first ──────────────────────────────────────
        # Clamp bbox to image bounds
        bx = max(0, bx)
        by = max(0, by)
        bw = min(bw, img_w - bx)
        bh = min(bh, img_h - by)

        # Crop face from full image
        face = img[by:by+bh, bx:bx+bw]
        if face.size == 0:
            face = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Resize cropped face to model input size
        face = cv2.resize(face, (self.img_size, self.img_size))

        # ── KEY FIX: Rescale landmarks relative to the crop ───────────────────
        lm_label = np.zeros(self.num_landmarks * 2, dtype=np.float32)
        mask     = np.zeros(self.num_landmarks * 2, dtype=np.float32)

        for feature_id, (lx, ly) in item['landmarks'].items():
            i = int(feature_id) - 1
            if 0 <= i < self.num_landmarks:
                # Shift landmark relative to crop origin, then normalize
                rel_x = (lx - bx) / bw if bw > 0 else 0
                rel_y = (ly - by) / bh if bh > 0 else 0

                # Only include landmarks that are inside the crop
                if 0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0:
                    lm_label[i*2]   = rel_x
                    lm_label[i*2+1] = rel_y
                    mask[i*2]       = 1.0
                    mask[i*2+1]     = 1.0

        if self.transform:
            face = self.transform(face)

        return face, torch.tensor(lm_label), torch.tensor(mask)

# ── 3. MODEL: Two separate models ─────────────────────────────────────────────

# Model A: Bounding box detector (takes full image)
class BBoxResNet(nn.Module):
    def __init__(self):
        super(BBoxResNet, self).__init__()
        resnet          = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone   = nn.Sequential(*list(resnet.children())[:-1])
        self.head       = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)   # x, y, w, h
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# Model B: Landmark detector (takes cropped face)
class LandmarkResNet(nn.Module):
    def __init__(self):
        super(LandmarkResNet, self).__init__()
        resnet          = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone   = nn.Sequential(*list(resnet.children())[:-1])
        self.head       = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 42)  # 21 landmarks x 2
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# ── 4. TRAINING ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    mse        = nn.MSELoss(reduction='none')

    for faces, landmarks, masks in tqdm(loader, desc="Training"):
        faces     = faces.to(device)
        landmarks = landmarks.to(device)
        masks     = masks.to(device)

        optimizer.zero_grad()
        pred_lm = model(faces)
        loss    = (mse(pred_lm, landmarks) * masks).sum() / (masks.sum() + 1e-8)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    mse        = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for faces, landmarks, masks in loader:
            faces     = faces.to(device)
            landmarks = landmarks.to(device)
            masks     = masks.to(device)
            pred_lm   = model(faces)
            loss      = (mse(pred_lm, landmarks) * masks).sum() / (masks.sum() + 1e-8)
            total_loss += loss.item()

    return total_loss / len(loader)

# ── 5. MAIN ───────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data       = load_data(DB_PATH)
    split      = int(0.8 * len(data))
    train_data = data[:split]
    val_data   = data[split:]

    train_dataset = AFLWDataset(train_data, IMAGES_PATH, IMG_SIZE, train_transform)
    val_dataset   = AFLWDataset(val_data,   IMAGES_PATH, IMG_SIZE, val_transform)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Train landmark model only (we already have bbox working)
    model     = LandmarkResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = validate(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_landmark_model.pth")
            print("  ✓ Saved best landmark model!")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.title("Landmark Model Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("landmark_training_loss.png")
    plt.show()
    print(f"Done! Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()


'''

## What Changed
The key difference is now during training each face is **cropped first** then landmarks are **rescaled relative to the crop** — so the model learns exactly where features are within a face, not within a full photo.

We also split into **two separate models**:
- `best_resnet_model.pth` — your existing bbox detector (already works!)
- `best_landmark_model.pth` — new landmark detector trained on crops

## Run It
```
& C:/Users/miamc/.conda/envs/aflw/python.exe c:/Users/miamc/Downloads/AFLW_folder/AFLW/Face_Detection_on_AFLW_dataset/src/cnn_training.py'''