import os
import sys
from datetime import datetime

sys.path.append("/mnt/aix22301/onj/code/")

import hydra
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

import ultralytics
from ultralytics.nn.modules.head import Classify, Detect  # add classify to layer number 8
from ultralytics.utils.loss import v8DetectionLoss, v8ClassificationLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from omegaconf import DictConfig
from einops import rearrange
from model2.modules.utils import preprocess_data
from attention import MultiHeadSelfAttention, MultiHeadCrossAttention, PatchEmbed3D, PatchEmbed2D

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
import time
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
    n_embed: int = 2048
    n_head: int = 8
    n_class: int = 2
    n_patch3d: tuple = (8, 8, 4)
    n_patch2d: tuple = (64, 64)
    width_2d: int = 2048
    width_3d: int = 512
    gpu: int = 7
    lambda1: float = 0.0  # det loss weight
    lambda2: float = 1.0  # cls loss weight
    epochs: int = 100
    lr: float = 1e-4
    batch: int = 16


class CTBackbone(nn.Module):
    def __init__(self):
        super(CTBackbone, self).__init__()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()


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
        "device": f"{Config.gpu}",
        "batch": 1,
        "lr0": 4e-2,
        "lrf": 3e-4,
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

    # self attention module
    mixer = MultiHeadSelfAttention(seq_len_x=seq_len_x, dim=Config.n_embed).to(trainer.device)  # 3D
    mixer = torch.compile(mixer)

    fusor = MultiHeadCrossAttention(seq_len_x=seq_len_x, seq_len_y=seq_len_y, dim=Config.n_embed).to(
        trainer.device
    )  # x -> y (3D -> 2D)
    fusor = torch.compile(fusor)

    # feature_expand = nn.Linear(Config.n_embed, Config.width_2d).to(trainer.device)
    # feature_expand = torch.compile(feature_expand)

    raw_model = trainer.model  # ultralytics.nn.tasks.DetectionModel
    raw_model = torch.compile(raw_model)

    classifier = Classify(c1=640, c2=2).to(trainer.device)  # ultralytics.nn.modules.head.Classify
    classifier = torch.compile(classifier)

    # # add parameters to the optimizer
    # trainer.optimizer.add_param_group({"params": classifier.parameters()})
    # trainer.optimizer.add_param_group(
    #     {"params": [param for module in [fusor, feature_expand, proj] for param in module.parameters()]}
    # )
    # trainer.optimizer.add_param_group({"params": [pe_x, pe_y]})

    # use adamw optimizer with cosine annealing
    # add classifier, raw_model, fusor, feature_expand, proj parameters
    optimizer = torch.optim.AdamW(
        [
            {"params": raw_model.parameters()},
            {"params": classifier.parameters()},
            {"params": [param for module in [mixer, fusor, proj] for param in module.parameters()]},
            {"params": [pe_x, pe_y]},
        ],
        lr=Config.lr,
    )

    models = [raw_model, classifier, mixer, fusor, proj]
    parameters = [pe_x, pe_y]

    # cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb * Config.epochs, eta_min=3e-4)
    patch_embed3d = PatchEmbed3D(patch_size=Config.n_patch3d, embed_dim=Config.n_embed).to(trainer.device)
    patch_embed2d = PatchEmbed2D(patch_size=Config.n_patch2d, embed_dim=Config.n_embed).to(trainer.device)

    register_hook(trainer.model, 8)

    # Set log_file
    log_dir = f"log/log_time_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_lr_{Config.lr}_batch_{Config.batch}_epochs_{Config.epochs}_lambda1_{Config.lambda1}_lambda2_{Config.lambda2}_patch3d_{Config.n_patch3d}_patch2d_{Config.n_patch2d}_embed_{Config.n_embed}_head_{Config.n_head}_width2d_{Config.width_2d}_width3d_{Config.width_3d}"
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
                trainer.loss.backward()

                # Gradient clipping
                for model in models:
                    clip_grad_norm_(model.parameters(), max_norm=1.0)

                for parameter in parameters:
                    clip_grad_norm_(parameter, max_norm=1.0)

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                with torch.no_grad():
                    preds = F.softmax(preds, dim=1)
                    pred = preds.argmax(dim=1).int().item()
                    mark = (pred == data["onj_cls"]).int().item()
                    log_message = f"Epoch: {epoch}, Step {(epoch*nb)+i+1}, Loss: {trainer.loss.item():.4f}, Mark: {mark}, Pred: {pred}"
                    pbar.set_description(log_message)

                    with open(log_file, "a") as f:
                        f.write(
                            f"{(epoch*nb)+i+1} train {trainer.loss.item():.4f} mark {mark} pred {pred} lr {scheduler.get_last_lr()}\n"
                        )


if __name__ == "__main__":
    main()
