"""
SimpleVision: CNN-Based Early Vision System
--------------------------------------------
Modes:
- TRAIN: self-supervised reconstruction on natural images
- RUN: real-time vision front-end (edges, motion, SDR)
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import random
from tqdm import tqdm

# ---------------------------
# Config
# ---------------------------
IMG_SIZE = 128
SDR_SIZE = 8192
SDR_SPARSITY = 0.03
TOP_K = int(SDR_SIZE * SDR_SPARSITY)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
TRAIN_DATA_PATH = "./train_images"  # folder of .jpg/.png files
SAVE_PATH = "./simplevision_weights.pt"

# ---------------------------
# Utility Functions
# ---------------------------
def load_images_from_folder(folder, size=IMG_SIZE, max_imgs=5000):
    """
    Load images from a specified folder and resize them.

    Args:
        folder (str): Path to the folder containing images.
        size (int): Size to resize images to (default: IMG_SIZE).
        max_imgs (int): Maximum number of images to load (default: 5000).

    Returns:
        list: List of loaded and resized images.
    """
    exts = ("*.jpg", "*.png", "*.jpeg")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = files[:max_imgs]
    imgs = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs

def preprocess_batch(batch):
    """
    Preprocess a batch of images for model input.

    Args:
        batch (list): List of images to preprocess.

    Returns:
        Tensor: Preprocessed batch as a PyTorch tensor.
    """
    batch = np.stack(batch).astype(np.float32) / 255.0
    batch = torch.from_numpy(batch).permute(0, 3, 1, 2)
    return batch.to(DEVICE)

def compute_sdr(features, sdr_size=SDR_SIZE, top_k=TOP_K):
    """
    Compute Sparse Distributed Representation (SDR) from features.

    Args:
        features (Tensor): Input feature tensor.
        sdr_size (int): Size of the SDR (default: SDR_SIZE).
        top_k (int): Number of top features to consider (default: TOP_K).

    Returns:
        numpy.ndarray: SDR as a binary array.
    """
    flat = features.flatten()
    sdr = torch.zeros(sdr_size, device=features.device)
    top_idx = torch.topk(flat, k=top_k, dim=0).indices % sdr_size
    sdr[top_idx] = 1
    return sdr.detach().cpu().numpy().astype(np.int8)

def random_warp(img, max_trans=0.08, max_rot=12.0, max_scale=0.10):
    """
    Apply a random affine warp to an image.

    Args:
        img (numpy.ndarray): Input image (HxWxC).
        max_trans (float): Maximum translation (default: 0.08).
        max_rot (float): Maximum rotation angle in degrees (default: 12.0).
        max_scale (float): Maximum scaling factor (default: 0.10).

    Returns:
        numpy.ndarray: Warped image.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    angle = random.uniform(-max_rot, max_rot)
    scale = 1.0 + random.uniform(-max_scale, max_scale)
    tx = random.uniform(-max_trans, max_trans) * w
    ty = random.uniform(-max_trans, max_trans) * h
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

def process_single_image(image):
    """
    Process a single RGB image and extract features.

    Args:
        image (numpy.ndarray): Input image (HxWxC).

    Returns:
        tuple: (f1, f2, f3, edges) where:
            - f1, f2, f3 are feature maps from each encoder layer (torch tensors)
            - edges is a numpy array of edge detection results
    """
    # Resize and preprocess
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    if image.shape[2] == 3 and image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # Load model if not already loaded
    if not hasattr(process_single_image, 'model'):
        process_single_image.model = SimpleVisionNet().to(DEVICE)
        process_single_image.model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        process_single_image.model.eval()
    
    # Get features
    with torch.no_grad():
        f1, f2, f3 = process_single_image.model.encoder(x)
        
        # Compute edges from f1
        edges = torch.sqrt(torch.sum(f1 * f1, dim=1, keepdim=True))
        edges = edges - edges.min()
        if edges.max() > 0:
            edges = edges / edges.max()
        edges = edges.squeeze().cpu().numpy()
    
    return f1, f2, f3, edges

# ---------------------------
# Model Definitions
# ---------------------------

class Encoder(nn.Module):
    """
    Encoder module for the SimpleVisionNet.

    Args:
        in_channels (int): Number of input channels (default: 3).
        base_dim (int): Base dimension for the first layer (default: 32).
    """
    def __init__(self, in_channels=3, base_dim=32):
        super().__init__()
        # Enhanced first layer with more edge-sensitive filters
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 5, padding=2),  # Larger kernel for better edge detection
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3

class Decoder(nn.Module):
    """
    Decoder module for the SimpleVisionNet.

    Args:
        base_dim (int): Base dimension for the first layer (default: 32).
        out_channels (int): Number of output channels (default: 3).
    """
    def __init__(self, base_dim=32, out_channels=3):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(base_dim, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        return torch.sigmoid(self.out(x))

class SimpleVisionNet(nn.Module):
    """
    SimpleVisionNet model combining encoder and decoder.

    Attributes:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        feature_proj (Conv2d): Feature projection layer.
        motion_head (Sequential): Motion detection head.
        prev_f1 (Tensor): Previous frame's features for motion detection.
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.feature_proj = nn.Conv2d(128, 64, 1)
        # Improved motion head
        self.motion_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # Takes concatenated current + previous features
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        self.prev_f1 = None  # Temporal memory

    def forward(self, x):
        f1, f2, f3 = self.encoder(x)
        recon = self.decoder(f3)
        feat = self.feature_proj(f3)
        
        # Motion detection using temporal information
        if self.prev_f1 is None:
            self.prev_f1 = f1.detach()
            motion_input = torch.cat([f1, f1], dim=1)  # First frame, duplicate features
        else:
            motion_input = torch.cat([f1, self.prev_f1], dim=1)
            self.prev_f1 = f1.detach()
        
        motion_pred = self.motion_head(motion_input)
        return recon, feat, motion_pred

# ---------------------------
# TRAIN MODE
# ---------------------------
def train_model():
    """
    Train the SimpleVisionNet model on the dataset.

    Loads images, initializes the model, and performs training.
    """
    try:
        print(f"[INFO] Loading data from {TRAIN_DATA_PATH} ...")
        imgs = load_images_from_folder(TRAIN_DATA_PATH)
        if len(imgs) == 0:
            print("[ERROR] No training images found. Put some in ./train_images/")
            return

        # Enable deterministic training for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = SimpleVisionNet().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        print("[INFO] Starting training (reconstruction + motion) ...")
        model.train()
        motion_weight = 5.0
        motion_threshold = 0.1

        # Configure multiprocessing for data loading
        if torch.cuda.is_available():
            torch.multiprocessing.set_start_method('spawn', force=True)

        for epoch in range(EPOCHS):
            random.shuffle(imgs)
            total_loss = 0.0
            recon_total = 0.0
            motion_total = 0.0
            
            # Use tqdm with dynamic progress bar
            progress = tqdm(range(0, len(imgs), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for i in progress:
                batch_imgs = imgs[i:i + BATCH_SIZE]
                try:
                    warped_imgs = [random_warp(im) for im in batch_imgs]
                    x = preprocess_batch(batch_imgs)
                    xw = preprocess_batch(warped_imgs)

                    optimizer.zero_grad()
                    
                    f1, f2, f3 = model.encoder(x)
                    f1w, f2w, f3w = model.encoder(xw)
                    recon = model.decoder(f3)
                    feat = model.feature_proj(f3)
                    
                    motion_input = torch.cat([f1, f1w], dim=1)
                    motion_pred = model.motion_head(motion_input)

                    motion_target = torch.abs(f1w - f1).mean(dim=1, keepdim=True)
                    motion_target = (motion_target > motion_threshold).float() * motion_target

                    recon_loss = criterion(recon, x)
                    motion_loss = F.l1_loss(motion_pred, motion_target)
                    
                    loss = recon_loss + motion_weight * motion_loss
                    loss.backward()
                    optimizer.step()

                    recon_total += recon_loss.item()
                    motion_total += motion_loss.item()
                    total_loss += loss.item()

                    # Update progress bar
                    progress.set_postfix({
                        'recon_loss': f'{recon_loss.item():.6f}',
                        'motion_loss': f'{motion_loss.item():.6f}'
                    })

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print("\n[WARNING] GPU OOM, skipping batch")
                        continue
                    else:
                        raise e

            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print(f"  Recon Loss: {recon_total/len(imgs):.6f}")
            print(f"  Motion Loss: {motion_total/len(imgs):.6f}")
            print(f"  Total Loss: {total_loss/len(imgs):.6f}")

            # Save checkpoint after each epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss/len(imgs),
            }
            torch.save(checkpoint, SAVE_PATH + f'.epoch{epoch+1}')

        # Save final model
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"[INFO] Saved trained weights to {SAVE_PATH}")

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        # Save current state
        torch.save(model.state_dict(), SAVE_PATH + '.interrupted')
        print(f"[INFO] Saved interrupted state to {SAVE_PATH}.interrupted")
    
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------
# RUN MODE (Real-time vision)
# ---------------------------
def visualize_features(features, normalize=True):
    """
    Visualize feature maps by averaging across channels.

    Args:
        features (Tensor): Input feature tensor.
        normalize (bool): Whether to normalize the output (default: True).

    Returns:
        numpy.ndarray: Visualized feature map.
    """
    vis = torch.mean(features, dim=1, keepdim=True)  # Average across channels
    if normalize:
        vis = vis - vis.min()
        if vis.max() > 0:
            vis = vis / vis.max()
    return (vis.squeeze().cpu().numpy() * 255).astype(np.uint8)

def run_model():
    """
    Run the SimpleVisionNet model in real-time mode using webcam input.

    Displays input, feature maps, edges, reconstruction, and motion.
    """
    print(f"[INFO] Loading trained model from {SAVE_PATH}")
    model = SimpleVisionNet().to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    cap = cv2.VideoCapture(0)
    prev_f1 = None
    
    print("[INFO] Running SimpleVision (learned motion) ... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        x = preprocess_batch([frame_rgb])
        with torch.no_grad():
            f1, f2, f3 = model.encoder(x)
            recon = model.decoder(f3)
            feat = model.feature_proj(f3)
            
            if prev_f1 is None:
                motion_input = torch.cat([f1, f1], dim=1)
            else:
                motion_input = torch.cat([f1, prev_f1], dim=1)
            prev_f1 = f1.detach()
            
            motion_pred = model.motion_head(motion_input)

        # Visualize feature maps
        f1_vis = visualize_features(f1)
        f2_vis = visualize_features(f2)
        f3_vis = visualize_features(f3)
        
        # Resize feature visualizations to match original size
        f1_vis = cv2.resize(f1_vis, (orig_w, orig_h))
        f2_vis = cv2.resize(f2_vis, (orig_w, orig_h))
        f3_vis = cv2.resize(f3_vis, (orig_w, orig_h))
        
        # Apply colormaps for better visualization
        f1_color = cv2.applyColorMap(f1_vis, cv2.COLORMAP_VIRIDIS)
        f2_color = cv2.applyColorMap(f2_vis, cv2.COLORMAP_PLASMA)
        f3_color = cv2.applyColorMap(f3_vis, cv2.COLORMAP_MAGMA)

        # Enhanced edge detection visualization
        edges = torch.sqrt(torch.sum(f1 * f1, dim=1, keepdim=True))
        edges = edges - edges.min()
        if edges.max() > 0:
            edges = edges / edges.max()
        edges_up = F.interpolate(edges, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        edges_np = (edges_up.squeeze().cpu().numpy() * 255).astype(np.uint8)
        edges_np = cv2.applyColorMap(edges_np, cv2.COLORMAP_BONE)

        # Motion visualization with temporal smoothing
        motion_up = F.interpolate(motion_pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        motion_np = motion_up.squeeze().cpu().numpy()
        motion_np = motion_np - motion_np.min()
        if motion_np.max() > 0:
            motion_np = motion_np / motion_np.max()
        motion_uint8 = (np.clip(motion_np, 0.0, 1.0) * 255).astype(np.uint8)
        motion_color = cv2.applyColorMap(motion_uint8, cv2.COLORMAP_HOT)
        
        # compute SDR
        sdr = compute_sdr(feat, SDR_SIZE, TOP_K)
        active = np.sum(sdr)

        # reconstruct image for display
        recon_img = recon.squeeze().permute(1, 2, 0).cpu().numpy()
        recon_img = (recon_img * 255).astype(np.uint8)
        recon_img = cv2.resize(recon_img, (orig_w, orig_h))
        recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)

        # ---- display ----
        cv2.imshow("Input", frame)
        cv2.imshow("f1 (first layer)", f1_color)
        cv2.imshow("f2 (middle layer)", f2_color)
        cv2.imshow("f3 (deep layer)", f3_color)
        cv2.imshow("Edges (f1 mean)", edges_np)
        cv2.imshow("Reconstruction", recon_img)
        cv2.imshow("Motion (learned)", motion_color)

        print(f"Active SDR bits: {active}/{SDR_SIZE}", end="\r")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

class VideoResult:
    """
    Container for video processing results.

    Args:
        f1_features (Tensor): First layer features (N, C, H, W).
        f2_features (Tensor): Second layer features (N, C, H, W).
        f3_features (Tensor): Third layer features (N, C, H, W).
        edges (Tensor): Edge detection results (N, H, W).
        motion (Tensor): Motion detection results (N, H, W).
    """
    def __init__(self, f1_features, f2_features, f3_features, edges, motion):
        """
        Container for video processing results
        Args:
            f1_features: First layer features (N, C, H, W)
            f2_features: Second layer features (N, C, H, W) 
            f3_features: Third layer features (N, C, H, W)
            edges: Edge detection results (N, H, W)
            motion: Motion detection results (N, H, W)
        where N is number of frames
        """
        self.f1 = f1_features
        self.f2 = f2_features
        self.f3 = f3_features
        self.edges = edges
        self.motion = motion

def process_video(video_path, max_frames=None):
    """
    Process a video file and return features for each frame.

    Args:
        video_path (str): Path to video file.
        max_frames (int): Maximum number of frames to process (optional).

    Returns:
        VideoResult: Object containing features, edges, and motion.
    """
    # Load model if not already loaded
    if not hasattr(process_video, 'model'):
        process_video.model = SimpleVisionNet().to(DEVICE)
        process_video.model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        process_video.model.eval()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    f1_list, f2_list, f3_list = [], [], []
    edges_list, motion_list = [], []
    frame_count = 0
    prev_f1 = None
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
                
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
            x = preprocess_batch([frame_resized])
            
            # Get features
            f1, f2, f3 = process_video.model.encoder(x)
            
            # Compute motion
            if prev_f1 is None:
                motion_input = torch.cat([f1, f1], dim=1)
            else:
                motion_input = torch.cat([f1, prev_f1], dim=1)
            prev_f1 = f1.detach()
            
            motion_pred = process_video.model.motion_head(motion_input)
            
            # Compute edges
            edges = torch.sqrt(torch.sum(f1 * f1, dim=1, keepdim=True))
            edges = edges - edges.min()
            if edges.max() > 0:
                edges = edges / edges.max()
            
            # Store results
            f1_list.append(f1.cpu())
            f2_list.append(f2.cpu())
            f3_list.append(f3.cpu())
            edges_list.append(edges.squeeze().cpu())
            motion_list.append(motion_pred.squeeze().cpu())
            
            frame_count += 1
    
    cap.release()
    
    # Stack all results
    f1_features = torch.cat(f1_list, dim=0)
    f2_features = torch.cat(f2_list, dim=0)
    f3_features = torch.cat(f3_list, dim=0)
    edges = torch.stack(edges_list)
    motion = torch.stack(motion_list)
    
    return VideoResult(f1_features, f2_features, f3_features, edges, motion)

def process_image_sequence(input_dir, max_frames=None):
    """
    Process a directory of sequential images and write out feature visualizations,
    motion maps and edge maps into the same directory.

    Args:
        input_dir (str): Path to folder containing sequential images.
        max_frames (int): Optional max number of frames to process.
    """
    # collect image files
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files = sorted(files)
    if len(files) == 0:
        print(f"[ERROR] No images found in {input_dir}")
        return
    if max_frames:
        files = files[:max_frames]

    # load model once
    model = SimpleVisionNet().to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    prev_f1 = None
    with torch.no_grad():
        for idx, fp in enumerate(files):
            img_bgr = cv2.imread(fp)
            if img_bgr is None:
                continue
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            x = preprocess_batch([img_resized])

            # forward pass
            f1, f2, f3 = model.encoder(x)
            recon = model.decoder(f3)
            feat = model.feature_proj(f3)

            # motion
            if prev_f1 is None:
                motion_input = torch.cat([f1, f1], dim=1)
            else:
                motion_input = torch.cat([f1, prev_f1], dim=1)
            prev_f1 = f1.detach()
            motion_pred = model.motion_head(motion_input)

            # edges (normalize and upsample)
            edges = torch.sqrt(torch.sum(f1 * f1, dim=1, keepdim=True))
            edges = edges - edges.min()
            if edges.max() > 0:
                edges = edges / edges.max()
            edges_up = F.interpolate(edges, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            edges_np = (edges_up.squeeze().cpu().numpy() * 255).astype(np.uint8)
            edges_color = cv2.applyColorMap(edges_np, cv2.COLORMAP_BONE)

            # motion upsample + color
            motion_up = F.interpolate(motion_pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            motion_np = motion_up.squeeze().cpu().numpy()
            motion_np = motion_np - motion_np.min()
            if motion_np.max() > 0:
                motion_np = motion_np / motion_np.max()
            motion_uint8 = (np.clip(motion_np, 0.0, 1.0) * 255).astype(np.uint8)
            motion_color = cv2.applyColorMap(motion_uint8, cv2.COLORMAP_HOT)

            # visualize features and upsample to original size
            f1_vis = visualize_features(f1)  # uint8 HxW
            f2_vis = visualize_features(f2)
            f3_vis = visualize_features(f3)

            f1_vis_up = cv2.resize(f1_vis, (orig_w, orig_h))
            f2_vis_up = cv2.resize(f2_vis, (orig_w, orig_h))
            f3_vis_up = cv2.resize(f3_vis, (orig_w, orig_h))

            f1_color = cv2.applyColorMap(f1_vis_up, cv2.COLORMAP_VIRIDIS)
            f2_color = cv2.applyColorMap(f2_vis_up, cv2.COLORMAP_PLASMA)
            f3_color = cv2.applyColorMap(f3_vis_up, cv2.COLORMAP_MAGMA)

            # reconstruction image
            recon_img = recon.squeeze().permute(1, 2, 0).cpu().numpy()
            recon_img = (recon_img * 255).astype(np.uint8)
            recon_img = cv2.resize(recon_img, (orig_w, orig_h))
            recon_bgr = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)

            # Save outputs into same directory with suffixes
            base, ext = os.path.splitext(fp)
            # use 4-digit index to avoid collisions if files have same name
            idx_str = f"{idx:04d}"
            cv2.imwrite(f"{base}_{idx_str}_f1.png", f1_color)
            cv2.imwrite(f"{base}_{idx_str}_f2.png", f2_color)
            cv2.imwrite(f"{base}_{idx_str}_f3.png", f3_color)
            cv2.imwrite(f"{base}_{idx_str}_edges.png", edges_color)
            cv2.imwrite(f"{base}_{idx_str}_motion.png", motion_color)
            cv2.imwrite(f"{base}_{idx_str}_recon.png", recon_bgr)

            print(f"[INFO] Saved visuals for frame {idx+1}/{len(files)} -> {base}_{idx_str}_*.png", end="\r")

    print(f"\n[INFO] Finished processing {len(files)} images in {input_dir}")

# ---------------------------
# Entry Point
# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "run", "process_dir"], default="run")
    parser.add_argument("--seq_dir", type=str, default=None, help="Directory of sequential images to process (used with --mode process_dir)")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frames to process for process_dir mode")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "run":
        run_model()
    elif args.mode == "process_dir":
        if not args.seq_dir:
            print("[ERROR] --seq_dir must be provided when mode is process_dir")
        else:
            process_image_sequence(args.seq_dir, max_frames=args.max_frames)
