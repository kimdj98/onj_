import os
import sys
import math
from datetime import datetime
import copy

sys.path.append("/mnt/aix22301/onj/code/")

import hydra
from dataclasses import dataclass
from dataclasses import asdict

import pandas as pd
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
from model3_feat.modules.utils import preprocess_data
from model3_feat.modules.transformer import (
    Transformer,
    PatchEmbed3D,
    PatchEmbed2D,
)  # MultiHeadSelfAttention, MultiHeadCrossAttention, PatchEmbed3D, PatchEmbed2D
from sklearn.metrics import roc_auc_score

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
    # NOTE: do not move this file into other folder for refactoring for example utils.py or etc.
    def __init__(self, log_file):
        self.log_file = log_file
        self.log_script_file = log_file.replace("log.txt", "log_script.py")
        self.log_data_file = log_file.replace("log.txt", "log_data.txt")
        with open(log_file, "w") as f:
            pass
        with open(self.log_script_file, "w") as f:
            pass
        with open(self.log_data_file, "w") as f:
            pass

        self.log_script()
        self.log_data()

        self.step = 0

    def log(self, message):
        self.step += 1
        with open(self.log_file, "a") as f:
            f.write(f"{self.step} " + message)

    def log_script(self):
        # Open current file and log every lines of code inside the file
        with open(__file__, "r") as f:
            lines = f.readlines()

        with open(self.log_script_file, "a") as f2:
            f2.writelines(lines)

    def log_data(self):
        with open(dataset_yaml, "r") as f:
            lines = f.readlines()

        with open(self.log_data_file, "a") as f:
            f.writelines(lines)

    def resume(self, resume_file):  # NOT USED
        # Open the existing log file to read its content
        with open("/".join(resume_file.split("/")[:-1]) + "/log.txt", "r") as f:
            lines = f.readlines()  # Read all lines

        # Write the content to the new log file
        with open(self.log_file, "w") as f2:
            f2.writelines(lines)

        # Get the last step number from the lines read
        if lines:
            self.step = int(lines[-1].split(" ")[0])


@dataclass
class Config:
    n_embed: int = 512
    n_head: int = 8
    n_class: int = 2
    n_layer: int = 2
    n_patch3d: tuple = (16, 16, 8)
    n_patch2d: tuple = (64, 64)
    width_2d: int = 1024
    width_3d: int = 512
    gpu: int = 6
    lambda1: float = 0.0  # det loss weight
    lambda2: float = 1.0  # cls loss weight
    epochs: int = 200
    lr: float = 3e-6
    batch: int = 1
    grad_accum_steps: int = 16 // batch
    eps: float = 1e-6
    resume: str = None  # set resume to None or path string
    # resume: str = (
    #     "/mnt/aix22301/onj/log/2024-08-07_12-32-58_lr_1e-06_gpu_7_layer_6_batch_16_epochs_200_patch3d_(16, 16, 8)_patch2d_(64, 64)_embed_1024_head_8_width2d_1024_width3d_512/best_auroc.pth"
    # )


class Classifier(nn.Module):
    def __init__(self, dim: int, seq_len: int, hidden_dim: int):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.fc3 = nn.Linear(seq_len, 1)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x).squeeze(-1)
        x = self.gelu(x)
        x = self.fc3(x)
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

        # self.common_embed_dim = model_config.common_embed_dim
        # CNN Feature Extractors
        # 2D CNN for 2D images
        self.cnn2d = nn.Sequential(
            # Initial convolutional layer
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            # Additional convolutional blocks with downsampling
            self._conv_block_2d(64, 128, downsample=True),
            self._conv_block_2d(128, 256, downsample=True),
            self._conv_block_2d(256, 512, downsample=True),
            # self._conv_block_2d(512, 512, downsample=True),  # TODO: remove this line if model is too small (optional)
        )

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
        dummy_input3d = torch.zeros(1, 1, model_config.width_3d, model_config.width_3d, 64)
        seq_len_x = (
            self.cnn3d(dummy_input3d).shape[2] * self.cnn3d(dummy_input3d).shape[3] * self.cnn3d(dummy_input3d).shape[4]
        )

        dummy_input2d = torch.zeros(1, 3, model_config.width_2d, model_config.width_2d)
        seq_len_y = self.cnn2d(dummy_input2d).shape[2] * self.cnn2d(dummy_input2d).shape[3]

        self.pe_x = nn.Parameter(
            torch.randn(1, seq_len_x, model_config.n_embed) * 1e-3
        )  # TODO: change 512 to model_config parameter
        self.pe_y = nn.Parameter(
            torch.randn(1, seq_len_y, model_config.n_embed) * 1e-3
        )  # TODO: change 512 to model_config parameter

        # self.proj3d = nn.Linear(model_config.common_embed_dim, model_config.n_embed)
        # self.proj2d = nn.Linear(model_config.common_embed_dim, model_config.n_embed)

        self.transformer = Transformer(
            n_layer=Config.n_layer, seq_len_x=seq_len_x, seq_len_y=seq_len_y, dim=model_config.n_embed, qk_scale=1.0
        )

        self.classifier = Classifier(model_config.n_embed, seq_len_y, model_config.n_embed * 4)

    def _conv_block_2d(self, in_channels, out_channels, downsample=False):
        stride = 2 if downsample else 1
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
        ]
        return nn.Sequential(*layers)

    def _conv_block_3d(self, in_channels, out_channels, downsample=False):
        stride = 2 if downsample else 1
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feature_x = self.cnn3d(x)
        feature_y = self.cnn2d(y)

        feature_x1 = rearrange(feature_x, "B C H W D -> B (H W D) C")
        feature_y1 = rearrange(feature_y, "B C H W -> B (H W) C")

        feature_x2 = feature_x1 + self.pe_x
        feature_y2 = feature_y1 + self.pe_y

        x1, y1 = self.transformer((feature_x2, feature_y2))

        out = self.classifier(y1)
        return out


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
    pth_dir = "/mnt/aix22301/onj/log/2024-11-06_15-16-18_n_embed_512_n_head_8_n_class_2_n_layer_2_n_patch3d_(16, 16, 8)_n_patch2d_(64, 64)_width_2d_1024_width_3d_512_gpu_6_lambda1_0.0_lambda2_1.0_epochs_200_lr_1e-05_batch_1_grad_accum_steps_16_eps_1e-06_resume_None/best_auroc.pth"
    checkpoint = torch.load(pth_dir, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device(f"cuda:{Config.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # test the model with validation set
    with torch.no_grad():
        model.eval()
        patients = []
        targets = []
        preds = []

        for k, data in enumerate(test_loader):
            data = trainer.preprocess_batch(data)
            proc_data = preprocess_data(base_path, data)

            patient_id = data["im_file"].split("/")[-1].split(".")[-2]

            if proc_data is None:
                print("No data: " + data["im_file"])
                continue

            pred = model(proc_data["CT_image"].float(), proc_data["img"].float())
            preds.append(round(F.sigmoid(pred.detach()).item(), 4))

            # binary cross entropy loss
            onj_cls = proc_data["onj_cls"].unsqueeze(0).unsqueeze(0).half()

            targets.append(onj_cls.item())
            patients.append(patient_id)

        pass
        # make dataframe for the results
        df = pd.DataFrame({"patient": patients, "target": targets, "pred": preds})

        # save the results to csv
        df.to_csv("/mnt/aix22301/onj/code/model3_feat/results.csv", index=False)


if __name__ == "__main__":
    main()
