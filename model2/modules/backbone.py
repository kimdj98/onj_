import os
import sys

sys.path.append("/mnt/aix22301/onj/code/")

import hydra
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import ultralytics
from ultralytics.nn.modules.head import Classify, Detect  # add classify to layer number 8
from ultralytics.utils.loss import v8DetectionLoss, v8ClassificationLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from omegaconf import DictConfig
from einops import rearrange
from model2.modules.utils import preprocess_data
from attention import MultiHeadCrossAttention, PatchEmbed3D, PatchEmbed2D

dataset_yaml = "/mnt/aix22301/onj/code/data/yolo_dataset.yaml"

version = "yolov8n"

args = {
    "task": "detect",
    "data": dataset_yaml,
    "imgsz": 640,
    "single_cls": False,
    "model": f"{version}.pt",
    "mode": "train",
}

import logging
import hydra
import torch
from torch.utils.data.dataset import ConcatDataset

from ultralytics import YOLO
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    TQDM,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
)


@dataclass
class Config:
    n_embed: int = 256
    n_head: int = 8
    n_class: int = 2
    n_patch3d: tuple = (8, 8, 4)
    n_patch2d: tuple = (16, 16)
    width_2d: int = 2048
    width_3d: int = 512

    lambda1: float = 0.5  # det loss weight
    lambda2: float = 0.5  # cls loss weight
    epochs: int = 10


@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main(cfg):
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

    custom_yaml = "/mnt/aix22301/onj/code/data/yolo_dataset.yaml"

    version = "yolov8x.pt"

    args = {
        "model": "/mnt/aix22301/onj/code/data/yolo_dataset.yaml",
        "imgsz": [1024, 1024],
        "task": "detect",
        "data": "/mnt/aix22301/onj/code/data/yolo_dataset.yaml",
        "mode": "train",
        "model": f"{version}",
        "device": "2",
        "batch": 1,
        "lr0": 1e-4,
        "lrf": 1e-3,
    }

    if torch.cuda.is_available():
        torch.cuda.current_device()  # HACK: Eagerly Initialize CUDA to avoid lazy initialization issue in _smart_load("trainer")

    input_tensor = torch.randn(4, 3, 224, 224)
    # # yolo_model = YOLO(model=dataset_yaml)
    yolo_model = YOLO(model=custom_yaml, task="detect", verbose=False)
    trainer = yolo_model._smart_load("trainer")(overrides=args, _callbacks=yolo_model.callbacks)
    trainer._setup_train(world_size=1)

    train_loader = trainer.train_loader
    train_dataset = train_loader.dataset

    test_loader = trainer.test_loader
    test_dataset = test_loader.dataset

    optimizer = trainer.optimizer

    # cls_loss_items = cls_loss.detach()
    # return cls_loss, cls_loss_items

    pbar = enumerate(train_loader)

    nb = len(train_loader)
    nw = max(round(trainer.args.warmup_epochs * nb), 100) if trainer.args.warmup_epochs > 0 else -1  # warmup iterations

    # calculate the seq_len_x and seq_len_y
    # select one data in pbar to configure the data size and types.
    data = next(pbar)[1]
    data = preprocess_data(base_path, data)
    B, _, H, W, D = data["CT_image"].shape
    seq_len_x = (H // Config.n_patch3d[0]) * (W // Config.n_patch3d[1]) * (D // Config.n_patch3d[2])
    B, _, H, W = data["img"].shape
    seq_len_y = (H // Config.n_patch2d[0]) * (W // Config.n_patch2d[1])

    pe_x = nn.Parameter(torch.randn(1, seq_len_x, Config.n_embed, device=trainer.device) * 1e-3)
    pe_y = nn.Parameter(torch.randn(1, seq_len_y, Config.n_embed, device=trainer.device) * 1e-3)
    proj = nn.Linear(Config.n_embed, Config.n_embed).to(trainer.device)
    proj = torch.compile(proj)

    fusor = MultiHeadCrossAttention(seq_len_x=seq_len_x, seq_len_y=seq_len_y, dim=256).to(
        trainer.device
    )  # x -> y (3D -> 2D)
    fusor = torch.compile(fusor)

    feature_expand = nn.Linear(256, Config.width_2d).to(trainer.device)
    feature_expand = torch.compile(feature_expand)

    raw_model = trainer.model  # ultralytics.nn.tasks.DetectionModel
    raw_model = torch.compile(raw_model)

    classifier = Classify(c1=640, c2=2).to(trainer.device)  # ultralytics.nn.modules.head.Classify
    classifier = torch.compile(classifier)

    # add parameters to the optimizer
    trainer.optimizer.add_param_group({"params": classifier.parameters()})
    trainer.optimizer.add_param_group(
        {"params": [param for module in [fusor, feature_expand, proj] for param in module.parameters()]}
    )
    trainer.optimizer.add_param_group({"params": [pe_x, pe_y]})

    # trainer.optimizer.add_param_group({"params": [feature_expand.parameters()]})

    patch_embed3d = PatchEmbed3D(patch_size=Config.n_patch3d).to(trainer.device)  # no parameters
    patch_embed2d = PatchEmbed2D(patch_size=Config.n_patch2d).to(trainer.device)

    register_hook(trainer.model, 8)

    # Set log_file
    log_dir = "log2"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:
        pass

    for epoch in range(Config.epochs):
        pbar = TQDM(enumerate(train_loader), total=nb)
        # change model to train mode
        raw_model.train()
        classifier.train()
        fusor.train()

        for i, data in pbar:
            with torch.cuda.amp.autocast(trainer.amp):
                data = trainer.preprocess_batch(data)
                data = preprocess_data(base_path, data)  # adds CT data and unify the data device
                if data is None:  # case when only PA exist
                    continue

                patches_3d = patch_embed3d(data["CT_image"])  # B E H W D (Batch, Embedding, Height, Width, Depth)
                patches_3d = rearrange(patches_3d, "B E H W D -> B (H W D) E")  # B (HWD) E
                patches_3d = proj(patches_3d)  # embedding
                patches_3d += pe_x

                patches_2d = patch_embed2d(data["img"][:, 0:1])  # B E H W (Batch, Embedding, Height, Width)
                patches_2d = rearrange(patches_2d, "B E H W -> B (H W) E")
                patches_2d = proj(patches_2d)  # embedding
                patches_2d += pe_y

                attn_out = fusor(patches_3d, patches_2d)
                attn_out = feature_expand(attn_out)
                data["img"] = attn_out.unsqueeze(0).expand(-1, 3, -1, -1)  # 1 H W -> B 3 H W

                # fusor
                trainer.loss, _ = raw_model.loss(data)

                # Forward pass
                preds = classifier(feature_map)
                cls_loss = F.cross_entropy(
                    preds,
                    F.one_hot(data["onj_cls"].to(torch.int64), num_classes=2).view(1, -1).float(),
                    reduction="mean",
                )

                # Backward pass
                trainer.loss = Config.lambda1 * trainer.loss + Config.lambda2 * cls_loss
                trainer.scaler.scale(trainer.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                trainer.optimizer_step()  # includes zero_grad()

                # Log the loss
                log_message = f"Epoch: {epoch}, Step {(epoch*nb)+i+1}, Loss: {trainer.loss.item():.4f}"
                pbar.set_description(log_message)

                with open(log_file, "a") as f:
                    f.write(f"{(epoch*nb)+i+1} train {trainer.loss.item():.4f}\n")


if __name__ == "__main__":
    main()
