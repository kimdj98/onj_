import os
import sys
import math
from datetime import datetime
import copy

sys.path.append("/mnt/aix22301/onj/code/")

import hydra
from dataclasses import dataclass
from dataclasses import asdict

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
    n_layer: int = 3
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
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "_".join(f"{key}_{value}" for key, value in config_dict.items())
    )

    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, "log.txt"))
    writer = SummaryWriter(f"{log_dir}/tensorboard")

    max_full_batch = (
        len(train_loader) // Config.grad_accum_steps
    )  # calculate how many times one epoch should repeat the batch
    last_accum_step = len(train_loader) % Config.grad_accum_steps  # handle the edge case

    # ================================================================
    #                     Resume from checkpoint
    # ================================================================
    if Config.resume:
        # logger.resume(Config.resume)

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

    # for gradient flow debugging
    def print_grad(name):
        def hook(grad):
            if grad.sum() == 0:
                print(f"Gradient for {name} = 0")
            else:
                print(f"Gradient for {name} = {grad.abs().mean()}")

        return hook

    # DEBUG: Check gradients
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         param.register_hook(print_grad(name))

    criterion = torch.nn.BCEWithLogitsLoss()

    while epoch <= Config.epochs:
        epoch += 1
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

                patient_id = data["im_file"][0].split("/")[-1].split(".")[-2]

                with torch.cuda.amp.autocast(trainer.amp):
                    data = trainer.preprocess_batch(data)
                    data = preprocess_data(base_path, data)  # adds CT data and unify the data device
                    if data is None:  # case when only PA exist
                        print("No data")
                        continue

                    # Forward pass
                    pred = model(data["CT_image"], data["img"])

                    preds.append(round(F.sigmoid(pred.detach()).item(), 4))
                    # binary cross entropy loss
                    onj_cls = data["onj_cls"].unsqueeze(0).unsqueeze(0).half()
                    cls_loss = criterion(pred, onj_cls)

                    # Backward pass
                    if i >= (len(train_loader) - last_accum_step):
                        loss = (cls_loss) / last_accum_step
                    else:
                        loss = (cls_loss) / Config.grad_accum_steps  # mean loss

                    loss_accum += loss.detach()
                    loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # log the gradients and parameter values to tensorboard to check the training process
            writer.add_histogram("gradients", norm, epoch * nb + i)

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
                proc_data = preprocess_data(base_path, data)

                if proc_data is None:
                    print("No data: " + data["im_file"])
                    continue

                pred = model(proc_data["CT_image"], proc_data["img"])
                preds.append(round(F.sigmoid(pred.detach()).item(), 4))

                # binary cross entropy loss
                onj_cls = proc_data["onj_cls"].unsqueeze(0).unsqueeze(0).half()
                targets.append(onj_cls.item())

                cls_loss = criterion(pred, onj_cls)

                loss_accum += cls_loss.detach()

            loss_accum /= len(test_loader)

            try:
                print(f"valid epoch: {epoch} step {(epoch*nb)+i+1} loss: {loss_accum.item():.4f}")
            except:
                pass
            logger.log(
                f"epoch {epoch} valid {loss_accum.item():.4f} auroc {roc_auc_score(y_true=targets, y_score=preds)}\n"
            )

            if best_loss > loss_accum.item():
                best_loss = loss_accum.item()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_auroc": best_auroc,
                        "best_loss": best_loss,
                    },
                    f"{log_dir}/best_loss.pth",
                )
                print(f"best_loss: {best_loss} saved")

            if best_auroc < roc_auc_score(y_true=targets, y_score=preds):
                best_auroc = roc_auc_score(y_true=targets, y_score=preds)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_auroc": best_auroc,
                        "best_loss": best_loss,
                    },
                    f"{log_dir}/best_auroc.pth",
                )
                print(f"best_auroc: {best_auroc} saved")

            # save the last model for the last epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auroc": best_auroc,
                    "best_loss": best_loss,
                },
                f"{log_dir}/last.pth",
            )


if __name__ == "__main__":
    main()
