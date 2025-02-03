import os
import sys
import math
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

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
from model5.modules.backbone import ResNet18_2D
from model5.modules.transformer import (
    Transformer,
    PatchEmbed3D,
    PatchEmbed2D,
)
from model5.modules.clinical_model import * # includes parameters and data path for clinical model
from model5.modules.post_processor import ImageFeatureExtractor, Classifier
from sklearn.metrics import roc_auc_score
from logger import Logger
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
    n_embed: int = 512
    n_head: int = 8
    n_class: int = 1
    n_layer: int = 2
    n_patch3d: tuple = (16, 16, 8)
    n_patch2d: tuple = (64, 64)
    width_2d: int = 1024
    width_3d: int = 512
    gpu: int = 0
    lambda1: float = 0.0  # det loss weight
    lambda2: float = 1.0  # cls loss weight
    epochs: int = 200
    lr: float = 1e-5
    batch: int = 1
    grad_accum_steps: int = 16 // batch
    eps: float = 1e-6
    # resume: str = None  # set resume to None or path string
    resume: str = ("/mnt/aix22301/onj/log/2025-01-20_16-49-48_n_embed_512_n_head_8_n_class_1_n_layer_2_n_patch3d_(16, 16, 8)_n_patch2d_(64, 64)_width_2d_1024_width_3d_512_gpu_7_lambda1_0.0_lambda2_1.0_epochs_200_lr_1e-05_batch_1_grad_accum_steps_16_eps_1e-06_resume_None/last.pth")

clinical_model = ClinicalModel(HPARAMS)  # HPARAMS is defined in clinical_model.py

class TransformerModel(nn.Module):
    def __init__(
        self,
        hydra_config: DictConfig,  # hydra config
        model_config: Config,  # model config
        data_example: dict,  # for model configuration depending on the data size and etc
    ):
        super(TransformerModel, self).__init__()
        self.base_path = hydra_config.data.data_dir
        self.patch_embed3d = PatchEmbed3D(
            patch_size=model_config.n_patch3d, embed_dim=model_config.n_embed
        )
        self.patch_embed2d = PatchEmbed2D(
            patch_size=model_config.n_patch2d, embed_dim=model_config.n_embed
        )

        # self.common_embed_dim = model_config.common_embed_dim
        # CNN Feature Extractors
        # 2D CNN for 2D images
        self.cnn2d = ResNet18_2D()

        # 3D CNN for 3D images
        self.cnn3d = nn.Sequential(
            # Initial convolutional layer
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            # Additional convolutional blocks with downsampling
            self._conv_block_3d(64, 128, downsample=True),
            self._conv_block_3d(128, 256, downsample=True),
            self._conv_block_3d(256, 512, downsample=True),
            # self._conv_block_3d(512, 512, downsample=True),  # TODO: remove this line if model is too small (optional)
        )

        # B, _, H, W, D = data["CT_image"].shape
        dummy_input3d = torch.zeros(
            1, 1, model_config.width_3d, model_config.width_3d, 64
        )
        seq_len_x = (
            self.cnn3d(dummy_input3d).shape[2]
            * self.cnn3d(dummy_input3d).shape[3]
            * self.cnn3d(dummy_input3d).shape[4]
        )

        dummy_input2d = torch.zeros(1, 3, model_config.width_2d, model_config.width_2d)
        seq_len_y = (
            self.cnn2d(dummy_input2d).shape[2] * self.cnn2d(dummy_input2d).shape[3]
        )

        self.pe_x = nn.Parameter(
            torch.randn(1, seq_len_x, model_config.n_embed) * 1e-3
        )  # TODO: change 512 to model_config parameter
        self.pe_y = nn.Parameter(
            torch.randn(1, seq_len_y, model_config.n_embed) * 1e-3
        )  # TODO: change 512 to model_config parameter

        self.transformer = Transformer(
            n_layer=Config.n_layer,
            seq_len_x=seq_len_x,
            seq_len_y=seq_len_y,
            dim=model_config.n_embed,
            qk_scale=1.0,
        )
        self.clinical_model = clinical_model

        self.image_feature_extractor = ImageFeatureExtractor(
            model_config.n_embed, seq_len_y, model_config.n_embed * 4
        )
        self.classifier = Classifier(
            seq_len_y + clinical_model.HPARAMS["u2"], model_config.n_class
        )

    def _conv_block_3d(self, in_channels, out_channels, downsample=False):
        stride = 2 if downsample else 1
        layers = [
            nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.ReLU(),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        feature_x = self.cnn3d(x)
        feature_y = self.cnn2d(y)

        feature_x1 = rearrange(feature_x, "B C H W D -> B (H W D) C")
        feature_y1 = rearrange(feature_y, "B C H W -> B (H W) C")

        feature_x2 = feature_x1 + self.pe_x
        feature_y2 = feature_y1 + self.pe_y

        x1, y1 = self.transformer((feature_x2, feature_y2))

        z1 = self.clinical_model(z)

        y2 = self.image_feature_extractor(y1)

        # concat the features y2([1, 4096]) and z1([225])
        z1 = z1.unsqueeze(0)
        concat_feature = torch.cat((y2, z1), dim=1)
        out = self.classifier(concat_feature)
        return out

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

    # Convert Config to a dictionary
    config_instance = Config()
    config_dict = asdict(config_instance)

    # Generate the log_dir by joining key-value pairs in config_dict
    log_dir = "log/{}_{}".format(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "_".join(f"{key}_{value}" for key, value in config_dict.items()),
    )

    # ================================================================
    #                     Resume from checkpoint
    # ================================================================
    if Config.resume:
        # Load checkpoint to CPU
        checkpoint = torch.load(Config.resume, map_location=torch.device("cpu"))

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

        best_auroc = checkpoint["best_auroc"]
        best_loss = checkpoint["best_loss"]
        epoch = checkpoint["epoch"]

    # ================================================================
    #                    Validation loop
    # ================================================================
    # test the model with validation set
    model.eval()

    target_layers = [model.cnn2d.layer4[1].conv1,
                     model.cnn2d.layer4[1].conv2]

    for k, data in enumerate(test_loader):
        if k == 14:
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

        pred_wo_img = model(proc_data["CT_image"].float(), torch.zeros_like(proc_data["img"]).float(), clinical_data.float())

        img_contribution = pred - pred_wo_img

        # print(f"Prediction for {patient_id}:                {pred.item():.4f}")
        # print(f"Prediction without image for {patient_id}:  {pred_wo_img.item():.4f}")
        # print(f"Image contribution for {patient_id}:        {img_contribution.item():.4f}")
        
        pred = F.sigmoid(pred)

        print(f"Probability for {patient_id}:               {pred.item():.4f}")
        print(f"Probability without image for {patient_id}: {F.sigmoid(pred_wo_img).item():.4f}")
        print(f"Image contribution for {patient_id}:        {(pred - F.sigmoid(pred_wo_img)).item():.4f}")

        model.zero_grad()
        pred.backward()

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
            heatmap = plt.get_cmap("jet")(cam_resized.numpy())[:, :, :3]  # shape [H, W, 3]
            heatmaps.append(heatmap)

        original_image = data["img"].squeeze(0)

        overlay = overlay_heatmaps_and_image(heatmaps, original_image, alpha=0.6)
        plt.imsave(f"heatmap_{patient_id}_last_layer4.png", overlay)
        print(f"Saved heatmap for {patient_id}")

if __name__ == "__main__":
    main()
