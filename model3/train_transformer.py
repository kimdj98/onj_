# Standard library imports
import os
import sys
import math
import logging
import time
from datetime import datetime
from dataclasses import dataclass

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from einops import rearrange
from omegaconf import DictConfig
from ultralytics import YOLO

# Hydra imports
import hydra

# Local imports
sys.path.append("/mnt/aix22301/onj/code/")
from model3.modules.utils import preprocess_data
from model3.utils import get_cls_pa_attention_map, visualize_cls_pa_attention_map, Logger
from model3.modules.transformer import TransformerModel, CustomTransformerEncoderLayer, CustomTransformerEncoder
from model3.modules.transformer import PatchEmbed3D, PatchEmbed2D
from model3.modules.classifier import Classifier
from utils import check_gradients, save_file

# Constants
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
    n_embed: int = 1024
    n_head: int = 8
    n_class: int = 2
    n_layer: int = 2
    n_patch3d: tuple = (16, 16, 8)
    n_patch2d: tuple = (32, 32)
    width_2d: int = 1024
    width_3d: int = 512
    gpu: int = 7
    lambda1: float = 0.0  # det loss weight
    lambda2: float = 1.0  # cls loss weight
    epochs: int = 100
    lr: float = 3e-6
    batch: int = 1
    grad_accum_steps: int = 16 // batch
    eps: float = 1e-6
    resume: str = None  # set resume to None or path string
    # resume: str = (
    #     "/mnt/aix22301/onj/log/2024-08-07_12-32-58_lr_1e-06_gpu_7_layer_6_batch_16_epochs_200_patch3d_(16, 16, 8)_patch2d_(64, 64)_embed_1024_head_8_width2d_1024_width3d_512/best_auroc.pth"
    # )

def setup_yolo_trainer():
    dataset_yaml = "/mnt/aix22301/onj/code/data/yolo_dataset3.yaml"
    version = "yolov8x.pt"
    args = {
        "model": dataset_yaml,
        "imgsz": [1024, 1024],
        "task": "detect",
        "data": dataset_yaml,
        "mode": "train",
        "device": f"{Config.gpu}",
        "batch": 1,
        "lr0": 4e-2,
        "lrf": 3e-4,
    }

    if torch.cuda.is_available():
        torch.cuda.current_device()

    yolo_model = YOLO(model=dataset_yaml, task="detect", verbose=False)
    trainer = yolo_model._smart_load("trainer")(overrides=args, _callbacks=yolo_model.callbacks)
    trainer._setup_train(world_size=1)
    return trainer

def get_lr_scheduler(nb, max_lr, min_lr, warmup_steps, max_steps):
    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    return get_lr

def setup_logging(config):
    log_dir = f"log/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_resume_{config.resume!=None}_lr_{config.lr}_gpu_{config.gpu}_layer_{config.n_layer}_batch_{config.grad_accum_steps}_epochs_{config.epochs}_patch3d_{config.n_patch3d}_patch2d_{config.n_patch2d}_embed_{config.n_embed}_head_{config.n_head}_width2d_{config.width_2d}_width3d_{config.width_3d}"
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, "log.txt"), "/mnt/aix22301/onj/code/data/yolo_dataset3.yaml")
    writer = SummaryWriter(f"{log_dir}/tensorboard")
    return log_dir, logger, writer

def load_checkpoint(model, optimizer, device, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
                
    return checkpoint["best_auroc"], checkpoint["best_loss"], checkpoint["epoch"]

def save_checkpoint(model, optimizer, epoch, best_auroc, best_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_auroc": best_auroc,
        "best_loss": best_loss,
    }, path)

def train_step(model, data, criterion, config, trainer, base_path):
    data = trainer.preprocess_batch(data)
    data = preprocess_data(base_path, data)
    if data is None:
        return None, None, None
        
    pred, attn_weights = model(data["CT_image"], data["img"])
    pred_prob = round(F.sigmoid(pred.detach()).item(), 4)
    
    onj_cls = data["onj_cls"].unsqueeze(0).unsqueeze(0).half()
    cls_loss = criterion(pred, onj_cls)
    
    return cls_loss, pred_prob, attn_weights

def validate_step(model, data, criterion, trainer, base_path, epoch, log_dir):
    data = trainer.preprocess_batch(data)
    proc_data = preprocess_data(base_path, data)
    
    if proc_data is None:
        print("No data: " + data["im_file"])
        return None, None, None
        
    pred, attn_weights = model(proc_data["CT_image"], proc_data["img"])
    pred_prob = round(F.sigmoid(pred.detach()).item(), 4)
    
    onj_cls = proc_data["onj_cls"].unsqueeze(0).unsqueeze(0).half()
    cls_loss = criterion(pred, onj_cls)
    
    if (epoch - 1) % 10 == 0:
        save_attention_map(data, attn_weights, proc_data, epoch, log_dir)
        
    return cls_loss, pred_prob, onj_cls.item()

def save_attention_map(data, attn_weights, proc_data, epoch, log_dir):
    patient_code = data["im_file"].split("/")[-1].split(".")[0]
    attention_map_dir = os.path.join("attention_map", log_dir.split("/")[-1])
    os.makedirs(attention_map_dir, exist_ok=True)
    save_path = os.path.join(attention_map_dir, f"epoch_{epoch-1}_patient_{patient_code}.png")
    visualize_cls_pa_attention_map(attn_weights[-1], proc_data["img"], Config, save_path=save_path)

@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg):
    trainer = setup_yolo_trainer()
    base_path = cfg.data.data_dir
    
    train_loader = trainer.train_loader
    test_loader = trainer.get_dataloader(trainer.testset, batch_size=1, rank=-1, mode="train")
    nb = len(train_loader)
    
    # Setup learning rate scheduler
    max_lr, min_lr = Config.lr, Config.lr * 0.1
    warmup_steps = Config.epochs * nb // 10
    max_steps = Config.epochs * nb
    get_lr = get_lr_scheduler(nb, max_lr, min_lr, warmup_steps, max_steps)
    
    # Initialize model and optimizer
    model = TransformerModel(cfg, Config, next(iter(train_loader)))
    optimizer = torch.optim.AdamW([{"params": model.parameters()}], lr=Config.lr)
    
    # Setup logging
    log_dir, logger, writer = setup_logging(Config)
    
    # Calculate batch sizes
    max_full_batch = len(train_loader) // Config.grad_accum_steps
    last_accum_step = len(train_loader) % Config.grad_accum_steps
    
    # Initialize training variables
    best_auroc, best_loss, epoch = 0.0, 1e6, 0
    
    # Resume from checkpoint if specified
    if Config.resume:
        best_auroc, best_loss, epoch = load_checkpoint(model, optimizer, trainer.device, Config.resume)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    SKIP_TRAINING = False
    
    while epoch <= Config.epochs:
        epoch += 1
        
        if not SKIP_TRAINING:
            model.train()
            model.to(trainer.device)
            pbar = iter(enumerate(train_loader))
            
            for j in range(max_full_batch + (last_accum_step != 0)):
                loss_accum, norm = 0.0, 0.0
                preds = []
                
                for micro_steps in range(Config.grad_accum_steps):
                    try:
                        i, data = next(pbar)
                    except StopIteration:
                        break
                        
                    with torch.cuda.amp.autocast(trainer.amp):
                        cls_loss, pred_prob, _ = train_step(model, data, criterion, Config, trainer, base_path)
                        if cls_loss is None:
                            continue
                            
                        preds.append(pred_prob)
                        loss = cls_loss / (last_accum_step if i >= (len(train_loader) - last_accum_step) else Config.grad_accum_steps)
                        
                        norm = norm + torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                        loss_accum += loss.detach()
                        loss.backward()
                
                # Update model
                norm = norm / Config.grad_accum_steps
                lr = get_lr((epoch * nb) + i)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()
                
                # Log progress
                log_message = f"train epoch: {epoch} step {(epoch*nb)+i+1} norm: {norm:.4f} loss: {loss_accum.item():.4f} lr: {lr:.10f}"
                print(preds)
                print(log_message)
                logger.log(f"train {loss_accum.item():.4f} norm {norm:.4f} lr {lr:.10f}\n")
        else:
            print(f"Skipping training for epoch {epoch} to test validation code.")
            
        # Validation
        with torch.cuda.amp.autocast(trainer.amp), torch.no_grad():
            model.eval()
            model.to(trainer.device)
            loss_accum = 0.0
            targets, preds = [], []
            
            for k, data in enumerate(test_loader):
                cls_loss, pred_prob, target = validate_step(model, data, criterion, trainer, base_path, epoch, log_dir)
                if cls_loss is not None:
                    loss_accum += cls_loss.detach()
                    preds.append(pred_prob)
                    targets.append(target)
                    
            loss_accum /= len(test_loader)
            current_auroc = roc_auc_score(y_true=targets, y_score=preds)
            
            try:
                print(f"valid epoch: {epoch} step {(epoch*nb)+i+1} loss: {loss_accum.item():.4f}")
            except:
                pass
                
            logger.log(f"epoch {epoch} valid {loss_accum.item():.4f} auroc {current_auroc}\n")
            
            # Save checkpoints
            if best_loss > loss_accum.item():
                best_loss = loss_accum.item()
                save_checkpoint(model, optimizer, epoch, best_auroc, best_loss, f"{log_dir}/best_loss.pth")
                print(f"best_loss: {best_loss} saved")
                
            if best_auroc < current_auroc:
                best_auroc = current_auroc
                save_checkpoint(model, optimizer, epoch, best_auroc, best_loss, f"{log_dir}/best_auroc.pth")
                print(f"best_auroc: {best_auroc} saved")
                
            # Save last checkpoint
            save_checkpoint(model, optimizer, epoch, best_auroc, best_loss, f"{log_dir}/last.pth")

if __name__ == "__main__":
    main()
