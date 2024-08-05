import os
import sys
import math
from datetime import datetime
import copy

sys.path.append("/mnt/aix22301/onj/code/")

import hydra
from dataclasses import dataclass

import torch
import torch.nn as nn

nn.Conv1d
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
from sklearn.metrics import roc_auc_score

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


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        with open(log_file, "w") as f:
            pass
        self.step = 0

    def log(self, message):
        with open(self.log_file, "a") as f:
            f.write(f"{self.step}" + message)


@dataclass
class Config:
    n_embed: int = 1024
    n_head: int = 8
    n_class: int = 2
    n_layer: int = 8
    n_patch3d: tuple = (16, 16, 8)
    n_patch2d: tuple = (64, 64)
    width_2d: int = 1024
    width_3d: int = 512
    gpu: int = 7
    lambda1: float = 0.0  # det loss weight
    lambda2: float = 1.0  # cls loss weight
    epochs: int = 100
    lr: float = 1e-6
    batch: int = 1
    grad_accum_steps: int = 16 // batch


class Classifier(nn.Module):
    def __init__(self, dim: int, seq_len: int, hidden_dim: int):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.fc3 = nn.Linear(seq_len, 1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x).squeeze(-1)
        x = self.gelu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        hydra_config: DictConfig,  # hydra config
        model_config: Config,  # model config
        data_example: dict,  # for model configuration depending on the data size and etc
    ):
        super(TransformerModel, self).__init__()
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

        self.proj3d = nn.Linear(model_config.n_embed, model_config.n_embed)
        self.proj2d = nn.Linear(model_config.n_embed, model_config.n_embed)

        self.transformer = Transformer(
            n_layer=Config.n_layer, seq_len_x=seq_len_x, seq_len_y=seq_len_y, dim=model_config.n_embed, qk_scale=1.0
        )

        self.classifier = Classifier(model_config.n_embed, seq_len_y, model_config.n_embed * 4)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        patches_3d = self.patch_embed3d(x)  # B E H W D (Batch, Embedding, Height, Width, Depth)
        patches_3d = rearrange(patches_3d, "B E H W D -> B (H W D) E")
        patches_3d = self.proj3d(patches_3d)  # 3d cubelet patch embedding
        patches_3d += self.pe_x

        patches_2d = self.patch_embed2d(y[:, 0:1])  # B E H W (Batch, Embedding, Height, Width)
        patches_2d = rearrange(patches_2d, "B E H W -> B (H W) E")
        patches_2d = self.proj2d(patches_2d)  # 2d square patch embedding
        patches_2d += self.pe_y

        x, y = self.transformer((patches_3d, patches_2d))

        out = self.classifier(y)
        return out


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

    test_loader = trainer.get_dataloader(trainer.testset, batch_size=1, rank=-1, mode="train")
    test_dataset = test_loader.dataset

    optimizer = trainer.optimizer

    pbar = enumerate(train_loader)

    nb = len(train_loader)

    max_lr = Config.lr
    min_lr = Config.lr * 0.1
    warmup_steps = Config.epochs * nb // 10
    max_steps = Config.epochs * nb

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    model = TransformerModel(cfg, Config, next(iter(train_loader)))
    # use AdamW optimizer with cosine annealing
    # add classifier, raw_model, fusor, feature_expand, proj parameters
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters()}],  # put raw_model inside the model
        lr=Config.lr,
    )

    # Set log_file
    log_dir = f"log/log_time_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_lr_{Config.lr}_batch_{Config.batch}_epochs_{Config.epochs}_patch3d_{Config.n_patch3d}_patch2d_{Config.n_patch2d}_embed_{Config.n_embed}_head_{Config.n_head}_width2d_{Config.width_2d}_width3d_{Config.width_3d}"
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, "log.txt"))

    max_full_batch = (
        len(train_loader) // Config.grad_accum_steps
    )  # calculate how many times one epoch should repeat the batch
    last_accum_step = len(train_loader) % Config.grad_accum_steps  # handle the edge case

    # for gradient flow debugging
    def print_grad(name):
        def hook(grad):
            if grad.sum() == 0:
                print(f"Gradient for {name} = 0")
            else:
                print(f"Gradient for {name} = {grad.abs().mean()}")

        return hook

    # # DEBUG: Check gradients
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         param.register_hook(print_grad(name))

    for epoch in range(Config.epochs):
        pbar = iter(enumerate(train_loader))
        # change model to train mode
        model.train()
        model.to(trainer.device)

        for j in range(max_full_batch + (last_accum_step != 0)):  # repeat for batch size + edge case
            # NOTE: comment/uncomment below to block/pass training code
            # continue
            loss_accum = 0.0
            preds = []
            for micro_steps in range(Config.grad_accum_steps):
                try:
                    i, data = next(pbar)
                except:
                    break  # should break to next epoch since j is already at the last batch

                with torch.cuda.amp.autocast(trainer.amp):
                    data = trainer.preprocess_batch(data)
                    data = preprocess_data(base_path, data)  # adds CT data and unify the data device
                    if data is None:  # case when only PA exist
                        print("No data")
                        continue

                    # Forward pass
                    pred = model(data["CT_image"], data["img"])
                    preds.append(round(pred.item(), 4))

                    # binary cross entropy loss
                    onj_cls = data["onj_cls"].unsqueeze(0).unsqueeze(0)
                    cls_loss = -(onj_cls * torch.log(pred) + (1 - onj_cls) * torch.log(1 - pred))

                    # clamp the loss to avoid nan
                    cls_loss = torch.clamp(cls_loss, 0, 100)

                    # Backward pass
                    if i >= (len(train_loader) - last_accum_step):
                        loss = (cls_loss) / last_accum_step
                    else:
                        loss = (cls_loss) / Config.grad_accum_steps  # mean loss

                    loss_accum += loss.detach()
                    loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            # DEBUG: Check gradients
            def check_gradients(named_parameters):
                for name, param in named_parameters:
                    if param.requires_grad:
                        if param.grad is None:
                            print(f"Parameter {name} has no gradient.")
                        else:
                            print(f"Parameter {name} gradient: {param.grad.abs().mean()}")

            # print(f"----------------  gradients  -------------------------")
            # check gradients for the first batch
            # if j == 0:
            #     check_gradients(model.named_parameters())
            # print(f"------------------------------------------------------")

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            lr = get_lr((epoch * nb) + i)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            log_message = f"train epoch: {epoch} step {(epoch*nb)+i+1} norm: {norm:.4f} loss: {loss_accum.item():.4f} lr: {lr:.10f}"
            print(log_message)
            print(preds)

            logger.log(f"train {loss_accum.item():.4f} norm {norm:.4f} lr {lr:.10f}\n")

        # test the model with validation set
        with torch.cuda.amp.autocast(trainer.amp), torch.no_grad():
            model.eval()
            loss_accum = 0.0
            targets = []
            preds = []

            for k, data in enumerate(test_loader):
                data = trainer.preprocess_batch(data)
                data = preprocess_data(base_path, data)

                if data is None:
                    print("No data")
                    continue

                pred = model(data["CT_image"], data["img"])
                preds.append(pred.item())

                # binary cross entropy loss
                onj_cls = data["onj_cls"].unsqueeze(0).unsqueeze(0)
                targets.append(onj_cls.item())

                cls_loss = -(onj_cls * torch.log(pred) + (1 - onj_cls) * torch.log(1 - pred))

                # clamp the loss to avoid nan
                cls_loss = torch.clamp(cls_loss, 0, 100)

                loss_accum += cls_loss.detach()

            loss_accum /= len(test_loader)

            print(f"valid epoch: {epoch} step {(epoch*nb)+i+1} loss: {loss_accum.item():.4f}")
            logger.log(
                f"epoch {epoch} valid {loss_accum.item():.4f} auroc {roc_auc_score(y_true=targets, y_score=preds)}\n"
            )


if __name__ == "__main__":
    main()
