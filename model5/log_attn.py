import os
import sys
import math
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
sys.path.append("/mnt/aix22301/onj/code/")

import hydra
from dataclasses import dataclass
from dataclasses import asdict

import torch
import torch.nn as nn

nn.Conv1d
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from omegaconf import DictConfig
from einops import rearrange
from model5.modules.utils import preprocess_data
from model5.modules.backbone import ResNet18_2D, resnet3d18
from model5.modules.transformer import (
    Transformer,
    PatchEmbed3D,
    PatchEmbed2D,
)

from model5.modules.clinical_model import *  # includes parameters and data path for clinical model
from model5.modules.post_processor import ImageFeatureExtractor, Classifier
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

dataset_yaml = "/mnt/aix22301/onj/code/data/yolo_dataset3.yaml"
version = "yolov8n"
args = {
    "task": "detect",
    "data": dataset_yaml,
    "imgsz": 640,
    "single_cls": False,
    "model": f"{version}.pt",
    "mode": "train",
}


@dataclass
class Config:
    debug: bool = False
    n_embed: int = 512
    n_head: int = 8
    n_class: int = 1
    n_layer: int = 2
    width_2d: int = 1024
    width_3d: int = 512
    gpu: int = 7
    lambda1: float = 0.0  # det loss weight
    lambda2: float = 1.0  #  cls loss weight
    epochs: int = 100
    lr: float = 1e-6
    batch: int = 1
    grad_accum_steps: int = 16 // batch
    grad_clip: int = 100
    grad_threshold: int = 150
    eps: float = 1e-6
    resume: str = (
        "/mnt/aix22301/onj/log/2025-02-14_23-32-50_debug_False_n_embed_512_n_head_8_n_class_1_n_layer_2_width_2d_1024_width_3d_512_gpu_7_lambda1_0.0_lambda2_1.0_epochs_100_lr_1e-06_batch_1_grad_accum_steps_16_grad_clip_100_grad_threshold_150_eps_1e-06_resume_None/best_auroc.pth"
    )


clinical_model = ClinicalModel(HPARAMS)  # HPARAMS is defined in clinical_model.py

from log_script import clinical_model
from log_script import TransformerModel


def overlay_heatmaps_and_image(heatmaps, original_image, alpha=0.6):
    """
    Combine multiple heatmaps into a single overlay on the original image.

    Args:
        heatmaps: List of heatmaps (each as a numpy array [H, W, 3]).
        original_image: Original input image tensor [3, H, W].
        alpha: Weight for the heatmap in the overlay.

    Returns:
        Final overlay image as a numpy array [H, W, 3].
    """
    # Combine heatmaps by averaging (normalized to avoid dominant layers)
    combined_heatmap = sum(heatmaps) / len(heatmaps)
    # combined_heatmap = normalize_heatmap(combined_heatmap)

    # Convert the original image to numpy format (scale to [0, 1])
    original_image_np = original_image.float().permute(1, 2, 0).cpu().numpy()
    # original_image_np = normalize_heatmap(original_image_np)

    # Overlay the heatmap on the original image
    overlay = (1 - alpha) * original_image_np + alpha * combined_heatmap
    
    # if overlay has > 1, clip it to 1
    overlay = np.clip(overlay, 0, 1)

    return overlay


def compute_rollout(attn_list):
    """
    attn_list: list of [B, num_heads, seq, seq] for each layer
    returns:   [B, seq, seq]
    """
    rollout = None
    for attn_map in attn_list:
        # average over heads => shape [B, seq, seq]
        attn_mean = attn_map.mean(dim=1)

        # If first layer, set rollout = attn_mean
        if rollout is None:
            rollout = attn_mean
        else:
            # Multiply them
            rollout = torch.bmm(attn_mean, rollout)  # shape [B, seq, seq]
    return rollout


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg):
    from torch.utils.tensorboard import SummaryWriter

    base_path = cfg.data.data_dir

    custom_yaml = "/mnt/aix22301/onj/code/data/yolo_dataset3.yaml"
    version = "yolov8x.pt"
    args = {
        "model": "/mnt/aix22301/onj/code/data/yolo_dataset3.yaml",
        "imgsz": [1024, 1024],
        "task": "detect",
        "data": "/mnt/aix22301/onj/code/data/yolo_dataset3.yaml",
        "mode": "train",
        "model": f"{version}",
        "device": f"{Config.gpu}",
        "batch": 1,
        "lr0": 4e-2,
        "lrf": 3e-4,
    }

    if torch.cuda.is_available():
        torch.cuda.current_device()  # HACK: Eagerly Initialize CUDA to avoid lazy initialization issue in _smart_load("trainer")

    yolo_model = YOLO(model=custom_yaml, task="detect", verbose=False)
    trainer = yolo_model._smart_load("trainer")(overrides=args, _callbacks=yolo_model.callbacks)
    trainer._setup_train(world_size=1)

    train_loader = trainer.train_loader
    train_dataset = train_loader.dataset

    test_loader = trainer.get_dataloader(trainer.testset, batch_size=1, rank=-1, mode="train")
    test_dataset = test_loader.dataset

    model = TransformerModel(cfg, Config, next(iter(train_loader)))
    # auroc 9.1666 (best model till 241113)
    pth_dir = Config.resume
    # pth_dir = None
    checkpoint = torch.load(pth_dir, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device(f"cuda:{Config.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # test the model with validation set
    model.eval()
    target_layers = [model.cnn2d.layer4[1].conv2]

    for k, data in enumerate(test_loader):
        if k == 30:
            break
        # Define hooks to capture the activation maps and gradients
        activations = list()
        gradients = list()
        forward_handles = list()
        backward_handles = list()

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # Register the hooks
        for target_layer in target_layers:
            forward_handles.append(target_layer.register_forward_hook(forward_hook))
            backward_handles.append(target_layer.register_backward_hook(backward_hook))

        data = trainer.preprocess_batch(data)
        proc_data = preprocess_data(base_path, data)

        patient_id = data["im_file"].split("/")[-1].split(".")[-2]
        idx = pt_CODE[pt_CODE["pt_CODE"] == patient_id]

        if idx.empty:
            print(f"{patient_id} has no clinical information")
            continue

        else:
            clinical_data = data_x.iloc[idx.index[0]]
            clinical_data = torch.tensor(clinical_data.values, dtype=torch.float32).to(trainer.device)

        if proc_data is None:
            print("No data: " + data["im_file"])
            continue

        pred = model(proc_data["CT_image"].float(), proc_data["img"].float(), clinical_data.float())
        pred = F.sigmoid(pred)
        
        # print the contribution of each modality (CT, PA, Clinical) to the prediction
        pred_CT = model(torch.zeros_like(proc_data["CT_image"]).float(), proc_data["img"].float(), clinical_data.float())
        pred_CT = F.sigmoid(pred_CT)
        pred_PA = model(proc_data["CT_image"].float(), torch.zeros_like(proc_data["img"]).float(), clinical_data.float())
        pred_PA = F.sigmoid(pred_PA)
        pred_clinical = model(proc_data["CT_image"].float(), proc_data["img"].float(), torch.zeros_like(clinical_data).float())
        pred_clinical = F.sigmoid(pred_clinical)
        
        print(f"CT: {(pred_CT-pred).item()}, PA: {(pred_PA-pred).item()}, Clinical: {(pred_clinical - pred).item()}")
        
        model.zero_grad()
        pred.backward()

        # Example for a 2-layer Transformer
        enc0_attn = model.transformer.encoder[0].latest_attn  # shape [B, num_heads, N, N]
        enc1_attn = model.transformer.encoder[1].latest_attn
        dec0_attn_s = model.transformer.decoder[0].latest_attn_s  # shape [B, num_heads, M, M]
        dec0_attn_c = model.transformer.decoder[0].latest_attn_c  # shape [B, num_heads, M, N]
        dec1_attn_s = model.transformer.decoder[1].latest_attn_s
        dec1_attn_c = model.transformer.decoder[1].latest_attn_c

        # For your 2-layer decoder's self-attn on y:
        attn_list_s = [dec0_attn_s]  # shape [B, num_heads, M, M]
        rollout_s = compute_rollout(attn_list_s)  # shape [B, M, M]

        # shape [B, M, M]
        # we just want a single importance score per patch => average over "query" dimension
        attn_per_patch = rollout_s.mean(dim=1)  # shape [B, M]

        # If seq_len_y = H * W
        H, W = 32, 32
        attn_map_2d = attn_per_patch[0].reshape(H, W)

        # Suppose attn_map_2d is shape [H_out, W_out] in the feature map space
        # Upsample to [original_H, original_W] = [1024, 1024]
        attn_t = torch.tensor(attn_map_2d, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # (1,1,H_out,W_out)
        
        # ====== Plot the attention distribution ======
        attn_values = attn_t.flatten().cpu().numpy()  # Convert to NumPy array

        # Define bins from 0 to 0.5 with a step of 0.05
        bins = torch.arange(0, 0.2, 0.001).numpy()  # Include 0.5 as the last bin edge

        # Plot histogram
        plt.hist(attn_values, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel('Attention Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Attention Values (0 to 0.2)')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f"histogram_{patient_id}_dec_attn2.png")
        plt.close()
        # ==============================================
        
        # delete attn_values greater than 0.03 to zero
        attn_t[attn_t > 0.03] = 0
        
        attn_upsampled = F.interpolate(attn_t, size=(1024, 1024), mode="bilinear", align_corners=False)
        # Assuming attn_t is your tensor
        attn_upsampled = attn_upsampled.squeeze(0).squeeze(0)  # shape [1024, 1024]

        # Convert to numpy, optionally colormap it
        attn_upsampled = attn_upsampled.cpu().numpy()
        attn_upsampled = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() + 1e-8) * 2
        attn_heatmap = plt.get_cmap("jet")(attn_upsampled)[:, :, :3]  # shape [1024, 1024, 3]

        # Remove the hooks
        for forward_handle, backward_handle in zip(forward_handles, backward_handles):
            forward_handle.remove()
            backward_handle.remove()

        heatmaps = list()
        for activation, grad_map in zip(activations, gradients):

            activation_map = activation.squeeze(0).detach().cpu()  # shape [C, H, W]
            grad_map = grad_map.detach().cpu()  # shape [C, H, W]

            # Global-average-pool the gradients over the spatial dimension
            alpha = grad_map.view(grad_map.size(0), -1).mean(dim=1)  # shape [C]

            # Weight each activation channel by alpha
            weighted_activations = activation_map * alpha[:, None, None] * 150
            cam = weighted_activations.sum(dim=0)

            # ReLU activation
            cam = F.relu(cam)

            # Normalize the CAM
            cam = (cam - cam.min()) / (cam.max() + 1e-8)

            cam_4d = cam.unsqueeze(0).unsqueeze(0)
            cam_resized = F.interpolate(cam_4d, size=(1024, 1024), mode="bilinear", align_corners=False)
            cam_resized = cam_resized.squeeze(0).squeeze(0)  # shape [1024, 1024]

            # Resize the CAM to the original image size
            cam_heatmap = plt.get_cmap("jet")(cam_resized.numpy())[:, :, :3]  # shape [H, W, 3]
            heatmaps.append(cam_heatmap)

        original_image = data["img"].squeeze(0)

        overlay = overlay_heatmaps_and_image([attn_heatmap], original_image, alpha=0.6)
        plt.imsave(f"heatmap_{patient_id}_dec_attn1.png", overlay)
        print(f"Saved heatmap_{patient_id}_dec_attn1.png")

        # combined_heatmap = cam_heatmap + attn_heatmap  # or do some weighting
        # # combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() + 1e-8)
        # # overlay = (1 - alpha) * original_image + alpha * combined_heatmap


if __name__ == "__main__":
    main()
