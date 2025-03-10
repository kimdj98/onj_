import os
import sys
import math
from datetime import datetime
import pandas as pd
import argparse
import matplotlib.pyplot as plt

sys.path.append("/mnt/aix22301/onj/code/")

import hydra
from dataclasses import dataclass
from dataclasses import asdict

import torch
import torch.nn as nn

nn.Conv1d
import torch.nn.functional as F

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

from log_script import *

clinical_model = ClinicalModel(HPARAMS)  # HPARAMS is defined in clinical_model.py

def normalize_heatmap(heatmap):
    """Normalize a heatmap to the range [0, 1]."""
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


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
    combined_heatmap = normalize_heatmap(combined_heatmap)

    # Convert the original image to numpy format (scale to [0, 1])
    original_image_np = original_image.float().permute(1, 2, 0).cpu().numpy()
    original_image_np = normalize_heatmap(original_image_np)

    # Overlay the heatmap on the original image
    overlay = (1 - alpha) * original_image_np + alpha * combined_heatmap

    return overlay


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg):
    from torch.utils.tensorboard import SummaryWriter

    base_path = cfg.data.data_dir
    best_auroc = 0.0
    best_loss = 1e6
    epoch = 0

    # Hook function to capture the output
    def hook_fn(module, input, output):
        global feature_map
        feature_map = output

    # Function to register the hook
    def register_hook(model, layer_index):
        global feature_map
        feature_map = None
        layer = list(model.model.children())[layer_index]
        layer.register_forward_hook(hook_fn)

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
    trainer = yolo_model._smart_load("trainer")(
        overrides=args, _callbacks=yolo_model.callbacks
    )
    trainer._setup_train(world_size=1)

    train_loader = trainer.train_loader
    train_dataset = train_loader.dataset

    test_loader = trainer.get_dataloader(
        trainer.testset, batch_size=1, rank=-1, mode="train"
    )
    test_dataset = test_loader.dataset

    criterion = torch.nn.BCEWithLogitsLoss()
    pbar = enumerate(train_loader)
    nb = len(train_loader)

    model = TransformerModel(cfg, Config, next(iter(train_loader)))
    # use AdamW optimizer with cosine annealing
    # add classifier, raw_model, fusor, feature_expand, proj parameters
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters()}],  # put raw_model inside the model
        lr=Config.lr,
    )

    # ================================================================
    #                     Resume from checkpoint
    # ================================================================
    if Config.resume:
        # Load checkpoint to CPU
        checkpoint = torch.load(Config.resume, map_location=torch.device("cpu"), weights_only=False)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Move model to the correct device
        model.to(trainer.device)

        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Move optimizer state to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(trainer.device)

    # ================================================================
    #                    Validation loop
    # ================================================================
    # test the model with validation set
    model.eval()

    target_layers = [
        # model.cnn2d.layer3[1].conv2,
        # model.cnn2d.layer4[0].conv1,
        # model.cnn2d.layer4[0].conv2,
        model.cnn2d.layer4[1].conv1, 
        model.cnn2d.layer4[1].conv2,
        ]


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
        
        pred = model(proc_data["CT_image"].float(), proc_data["img"].float(), clinical_data.float())[2]

        # pred_wo_img = model(proc_data["CT_image"].float(), torch.zeros_like(proc_data["img"]).float(), clinical_data.float())[0]

        # img_contribution = pred - pred_wo_img

        # print(f"Prediction for {patient_id}:                {pred.item():.4f}")
        # print(f"Prediction without image for {patient_id}:  {pred_wo_img.item():.4f}")
        # print(f"Image contribution for {patient_id}:        {img_contribution.item():.4f}")
        
        pred = F.sigmoid(pred)

        # print(f"Probability w/ image for {patient_id}: {pred.item():.4f}")
        # print(f"Probability wo image for {patient_id}: {F.sigmoid(pred_wo_img).item():.4f}")
        # print(f"Image contribution for {patient_id}       : {(pred - F.sigmoid(pred_wo_img)).item():.4f}")
        # print(f"Image contribution for {patient_id}(logit): {(torch.log(pred/(1-pred)) - pred_wo_img).item():.4f}")

        model.zero_grad()
        pred.backward()

        # Remove the hooks
        for forward_handle, backward_handle in zip(forward_handles, backward_handles):
            forward_handle.remove()
            backward_handle.remove()

        heatmaps = list()
        gradients.reverse()
        for activation, grad_map in zip(activations, gradients):
            grad_map = grad_map.squeeze(0)  # shape [C, H, W]
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
            cam[:,:cam.size(1)//6]=0
            cam[:,-cam.size(1)//6:]=0
            
            cam[:cam.size(0)//4,:] = 0
            cam[-cam.size(0)//4:,:] = 0
            
            cam = (cam) / (0.01 + 1e-8)
            
            # Clamp the CAM to the range [0, 1]
            cam = torch.clamp(cam, 0, 1)
            

            cam_4d = cam.unsqueeze(0).unsqueeze(0)
            cam_resized = F.interpolate(cam_4d, size=(1024, 1024), mode="bilinear", align_corners=False)
            cam_resized = cam_resized.squeeze(0).squeeze(0)  # shape [1024, 1024]

            # Resize the CAM to the original image size
            heatmap = plt.get_cmap("jet")(cam_resized.numpy())[:, :, :3]  # shape [H, W, 3]
            heatmaps.append(heatmap)

        original_image = data["img"].squeeze(0)

        overlay = overlay_heatmaps_and_image(heatmaps, original_image, alpha=0.6)
        plt.imsave(f"heatmap_{patient_id}_last_layer4.png", overlay)
        print(f"Saved heatmap for {patient_id}")

if __name__ == "__main__":
    main()
