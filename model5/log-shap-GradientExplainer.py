import sys
import numpy as np

sys.path.append("/mnt/aix22301/onj/code/")

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

import shap

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

class Model2DWrapper(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.model = full_model
        # We'll store the real 3D & clinical data in attributes
        self.real_3d = None
        self.real_clinical = None

    def set_real_data(self, x_3d: torch.Tensor, x_clin: torch.Tensor): # to get the real 3d and clincal data
        """
        x_3d: shape [1, C3, D, H3, W3]   (whatever your 3D shape is)
        x_clin: shape [1, clinical_dim]
        """
        self.real_3d = x_3d
        self.real_clinical = x_clin

    def forward(self, x_2d: torch.Tensor):
        """
        x_2d: shape [B, C2, H2, W2]
        We pass the stored real 3D & clinical data to the main model.
        """
        # If real_3d or real_clinical is None, raise an error or handle it
        if self.real_3d is None or self.real_clinical is None:
            raise ValueError("Must call set_real_data() before forward!")
        
        # If you have B images in x_2d, replicate the single real_3d or real_clinical
        # if needed. Or assume B=1. For example:
        out = self.model(
            self.real_3d.expand(x_2d.shape[0], -1, -1, -1, -1),
            x_2d,
            self.real_clinical.expand(x_2d.shape[0], -1)
        )[0]
        return torch.sigmoid(out)

@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg):
    from torch.utils.tensorboard import SummaryWriter

    base_path = cfg.data.data_dir

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
    trainer = yolo_model._smart_load("trainer")(overrides=args, _callbacks=yolo_model.callbacks)
    trainer._setup_train(world_size=1)

    train_loader = trainer.train_loader
    train_dataset = train_loader.dataset

    test_loader = trainer.get_dataloader(trainer.testset, batch_size=1, rank=-1, mode="train")
    test_dataset = test_loader.dataset

    model = TransformerModel(cfg, Config, next(iter(train_loader)))
    # auroc 0.91666 (best model till 241113)
    pth_dir = "/mnt/aix22301/onj/log/2025-03-06_15-21-25_gpu_4_n_embed_256_n_layer_6_expansion_4_lr_1e-05_debug_False_n_head_8_n_class_1_width_2d_1024_width_3d_512_lambda1_0.0_lambda2_1.0_epochs_100_batch_1_grad_accum_steps_16_grad_clip_10000_grad_threshold_60_eps_1e-06_resume_None/best_auroc.pth"
    # pth_dir = None
    checkpoint = torch.load(pth_dir, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device(f"cuda:{Config.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def model_predict(img2d: np.ndarray) -> torch.Tensor:
        img2d_tensor = torch.tensor(img2d, dtype=torch.float32).to(device)

        img3d_tensor = None
        clinical_data = None

        model.eval()

        output = model(img3d_tensor, img2d_tensor, clinical_data)
        oiutput = F.sigmoid(output)

        return output

    # test the model with validation set
    model.eval()
    test_images = []
    targets = []
    preds = []
    patients = []

    for k, data in enumerate(test_loader):
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

        if k == 0:
            bg_2d = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32).to(device)
            wrapped_model = Model2DWrapper(model).eval().to(device)
            wrapped_model.set_real_data(proc_data["CT_image"].to(torch.float32), clinical_data.to(torch.float32))
            explainer = shap.DeepExplainer(wrapped_model, bg_2d)

        if k == 10:
            break

        wrapped_model.set_real_data(proc_data["CT_image"].to(torch.float32), clinical_data.to(torch.float32))

        if proc_data is None:
            print("No data: " + data["im_file"])
            continue

        pred = model(proc_data["CT_image"].float(), proc_data["img"].float(), clinical_data.float())[0]
        preds.append(round(F.sigmoid(pred.detach()).item(), 4))

        # binary cross entropy loss
        onj_cls = proc_data["onj_cls"].unsqueeze(0).unsqueeze(0).half()
        targets.append(onj_cls.item())
        patients.append(patient_id)
        model.to(torch.float32)

        shap_values = explainer.shap_values(proc_data["img"].to(device).to(torch.float32))

    pass


if __name__ == "__main__":
    main()
