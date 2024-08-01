import os
import sys
from datetime import datetime
import copy

sys.path.append("/mnt/aix22301/onj/code/")

import hydra
from dataclasses import dataclass

import torch
import torch.nn as nn

nn.Transformer
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
from model2.modules.transformer import (
    Transformer,
    PatchEmbed3D,
    PatchEmbed2D,
)  # MultiHeadSelfAttention, MultiHeadCrossAttention, PatchEmbed3D, PatchEmbed2D

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
    n_embed: int = 1024
    n_head: int = 8
    n_class: int = 2
    n_layer: int = 2
    n_patch3d: tuple = (16, 16, 8)
    n_patch2d: tuple = (64, 64)
    width_2d: int = 1024
    width_3d: int = 512
    gpu: int = 5
    lambda1: float = 0.0  # det loss weight
    lambda2: float = 1.0  # cls loss weight
    epochs: int = 100
    lr: float = 3e-4
    batch: int = 1
    grad_accum_steps: int = 12 // batch


class CTBackbone(nn.Module):
    def __init__(self):
        super(CTBackbone, self).__init__()

    def forward(self, x):
        return x


class FusionModel(nn.Module):
    def __init__(
        self,
        hydra_config: DictConfig,  # hydra config
        model_config: Config,  # model config
        data_example: dict,  # for model configuration depending on the data size and etc
        trainer: DetectionTrainer,
        ct_backbone: CTBackbone,
        yolo: YOLO,
    ):
        super(FusionModel, self).__init__()
        self.base_path = hydra_config.data.data_dir
        self.patch_embed3d = PatchEmbed3D(patch_size=model_config.n_patch3d, embed_dim=model_config.n_embed)
        self.patch_embed2d = PatchEmbed2D(patch_size=model_config.n_patch2d, embed_dim=model_config.n_embed)

        # to configure the data size and types
        data = preprocess_data(self.base_path, data_example)

        B, _, H, W, D = data["CT_image"].shape
        seq_len_x = (
            (H // model_config.n_patch3d[0]) * (W // model_config.n_patch3d[1]) * (D // model_config.n_patch3d[2])
        )

        B, _, H, W = data["img"].shape
        seq_len_y = (H // model_config.n_patch2d[0]) * (W // model_config.n_patch2d[1])

        self.pe_x = nn.Parameter(torch.randn(1, seq_len_x, model_config.n_embed) * 1e-3)
        self.pe_y = nn.Parameter(torch.randn(1, seq_len_y, model_config.n_embed) * 1e-3)

        self.proj = nn.Linear(model_config.n_embed, model_config.n_embed)
        self.transformer = Transformer(
            n_layer=Config.n_layer, seq_len_x=seq_len_x, seq_len_y=seq_len_y, dim=model_config.n_embed
        )

        # self.feature_expand = nn.Linear(Config.n_embed, Config.width_2d).to(trainer.device)
        # self.ct_backbone = ct_backbone  # for feature-level fusion

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        patches_3d = self.patch_embed3d(x)  # B E H W D (Batch, Embedding, Height, Width, Depth)
        patches_3d = rearrange(patches_3d, "B E H W D -> B (H W D) E")
        patches_3d = self.proj(patches_3d)  # embedding
        patches_3d += self.pe_x

        patches_2d = self.patch_embed2d(y[:, 0:1])  # B E H W (Batch, Embedding, Height, Width)
        patches_2d = rearrange(patches_2d, "B E H W -> B (H W) E")
        patches_2d = self.proj(patches_2d)  # embedding
        patches_2d += self.pe_y

        x, y = self.transformer((patches_3d, patches_2d))

        return y


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
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

    yolo_model = YOLO(model=custom_yaml, task="detect", verbose=False)
    trainer = yolo_model._smart_load("trainer")(overrides=args, _callbacks=yolo_model.callbacks)
    trainer._setup_train(world_size=1)

    train_loader = trainer.train_loader
    train_dataset = train_loader.dataset

    test_loader = trainer.test_loader
    test_dataset = test_loader.dataset

    optimizer = trainer.optimizer

    pbar = enumerate(train_loader)

    nb = len(train_loader)
    nw = max(round(trainer.args.warmup_epochs * nb), 100) if trainer.args.warmup_epochs > 0 else -1  # warmup iterations

    raw_model = trainer.model  # ultralytics.nn.tasks.DetectionModel
    register_hook(raw_model, 8)

    model = FusionModel(cfg, Config, next(iter(train_loader)), trainer, ct_backbone=None, yolo=raw_model)
    # use AdamW optimizer with cosine annealing
    # add classifier, raw_model, fusor, feature_expand, proj parameters
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters()}],  # put raw_model inside the model
        lr=Config.lr,
    )

    # cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb * Config.epochs, eta_min=3e-4)

    # Set log_file
    log_dir = f"log/log_time_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_lr_{Config.lr}_batch_{Config.batch}_epochs_{Config.epochs}_lambda1_{Config.lambda1}_lambda2_{Config.lambda2}_patch3d_{Config.n_patch3d}_patch2d_{Config.n_patch2d}_embed_{Config.n_embed}_head_{Config.n_head}_width2d_{Config.width_2d}_width3d_{Config.width_3d}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:
        pass

    max_full_batch = (
        len(train_loader) // Config.grad_accum_steps
    )  # calculate how many times one epoch should repeat the batch
    last_accum_step = len(train_loader) % Config.grad_accum_steps  # handle the edge case

    for epoch in range(Config.epochs):
        pbar = iter(enumerate(train_loader))
        # change model to train mode
        model.train()
        model.to(trainer.device)

        for j in range(max_full_batch + (last_accum_step != 0)):  # repeat for batch size + edge case
            loss_accum = 0.0
            for micro_steps in range(Config.grad_accum_steps):
                try:
                    i, data = next(pbar)
                except:
                    break  # should break to next epoch since j is already at the last batch

                with torch.cuda.amp.autocast(trainer.amp):

                    data = trainer.preprocess_batch(data)
                    data = preprocess_data(base_path, data)  # adds CT data and unify the data device
                    if data is None:  # case when only PA exist
                        continue

                    # Forward pass
                    data["img"] = model(data["CT_image"], data["img"])

                    det_loss, _ = raw_model.loss(data)
                    preds = model.classifier(feature_map)

                    data["onj_cls"] = torch.tensor(
                        data["onj_cls"], dtype=torch.int64, device=trainer.device
                    )  # HACK: to avoid error

                    cls_loss = F.cross_entropy(
                        preds,
                        F.one_hot(data["onj_cls"], num_classes=2).view(1, -1).float(),
                        reduction="mean",
                    )

                    # Backward pass
                    if i >= (len(train_loader) - last_accum_step):
                        loss = (Config.lambda1 * det_loss + Config.lambda2 * cls_loss) / last_accum_step
                    else:
                        loss = (Config.lambda1 * det_loss + Config.lambda2 * cls_loss) / Config.grad_accum_steps

                    loss_accum += loss.detach()
                    loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=150.0)

            # DEBUG: Check gradients
            def check_gradients(named_parameters):
                for name, param in named_parameters:
                    if param.requires_grad:
                        if param.grad is None:
                            print(f"Parameter {name} has no gradient.")
                        else:
                            print(f"Parameter {name} gradient: {param.grad.abs().mean()}")

            # Inside the training loop
            check_gradients(model.named_parameters())

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            with torch.no_grad():
                # preds = F.softmax(preds, dim=1)
                # pred = preds.argmax(dim=1).int().item()
                # mark = (pred == data["onj_cls"]).int().item()
                # log_message = (
                #     f"Epoch: {epoch}, Step {(epoch*nb)+i+1}, Loss: {loss_accum.item():.4f}, Mark: {mark}, Pred: {pred}"
                # )
                log_message = f"epoch: {epoch} step {(epoch*nb)+i+1} norm: {norm:.4f} loss: {loss_accum.item():.4f}"
                print(log_message)

                # with open(log_file, "a") as f:
                #     f.write(
                #         f"{(epoch*nb)+i+1} train {loss_accum.item():.4f} mark {mark} pred {pred} lr {scheduler.get_last_lr()}\n"
                #     )

                with open(log_file, "a") as f:
                    f.write(
                        f"{(epoch*nb)+i+1} train {loss_accum.item():.4f} norm {norm:.4f} lr {scheduler.get_last_lr()[0]}\n"
                    )

    model.eval()


if __name__ == "__main__":
    main()
