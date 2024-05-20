# utils/training.py
from tqdm import tqdm
import torch
from torch.nn import functional as F


def train_one_epoch(model, dataloader, loss_fn, optimizer, auroc, acc, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader):
        batch["CT_image"] = batch["CT_image"].to(device)
        batch["CT_label"] = batch["CT_label"].to(device)
        batch["PA_image"] = batch["PA_image"].to(device)
        batch["PA_label"] = batch["PA_label"].to(device)

        optimizer.zero_grad()
        output = model(batch)
        loss_value = loss_fn(output, batch["CT_label"])
        loss_value.backward()
        optimizer.step()

        running_loss += loss_value.item()

        p = F.softmax(output, dim=1)
        auroc.update(p[:, 1], batch["CT_label"])
        acc.update(p[:, 1], batch["CT_label"])

    return running_loss / len(dataloader), auroc.compute().item(), acc.compute().item()


def validate(model, dataloader, loss_fn, auroc, acc, device):
    model.eval()
    running_loss = 0.0
    true_labels, predictions = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch["CT_image"] = batch["CT_image"].to(device)
            batch["CT_label"] = batch["CT_label"].to(device)
            output = model(batch)
            loss_value = loss_fn(output, batch["CT_label"])
            running_loss += loss_value.item()

            p = F.softmax(output, dim=1)
            auroc.update(p[:, 1], batch["CT_label"])
            acc.update(p[:, 1], batch["CT_label"])

            true_labels.extend(batch["CT_label"].cpu().numpy())
            predictions.extend(p[:, 1].cpu().numpy())

    return running_loss / len(dataloader), auroc.compute().item(), acc.compute().item(), true_labels, predictions
