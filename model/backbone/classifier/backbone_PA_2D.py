# backbone_PA.py
# 3 model will be implemented
#
# 1. YOLO(freezed) -> classifier
# 2. YOLO(not freezed) -> classifier
# 3. ResNET -> classifier

import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig
import ultralytics
import wandb

# from ultralytics.yolo.utils.ops import non_max_suppression
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO

# print colored text using colorama
import colorama

from PIL import Image, ImageDraw

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from tqdm import tqdm

# import wandb


class ClassifierModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierModel, self).__init__()
        # Convolutional layers to reduce channel dimensions
        # self.conv0 = nn.Conv2d(192 + 384 + 576, 128, kernel_size=1)  # Reduce channels of yolo_out_15, yolo_out_18, yolo_out_21
        self.conv1 = nn.Conv2d(64, 32, kernel_size=1)  # Reduce channels of yolo_out_15
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)  # Reduce channels of yolo_out_18
        self.conv3 = nn.Conv2d(256, 128, kernel_size=1)  # Reduce channels of yolo_out_21

        # Assuming you've resized your features to a common size, e.g., [2, 128, 128, 64]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 32))

        self.conv4 = nn.Conv2d(32 + 64 + 128, 32, kernel_size=1)
        self.conv4.register_forward_hook(self.hook_fn)

        self.fc1 = nn.Linear(32 * 64 * 32, 512)
        # self.dropout1 = nn.Dropout(p=0.5)  # Dropout layer after fc1
        self.fc2 = nn.Linear(512, 512)
        # self.dropout2 = nn.Dropout(p=0.5)  # Dropout layer after fc2
        self.fc3 = nn.Linear(512, num_classes)
        self.fc3.register_forward_hook(self.hook_fn2)

        # initialize fc layers with xavier uniform
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def hook_fn(self, module, input, output):
        self.f = output

    def hook_fn2(self, module, input, output):
        self.out = output

    def forward(self, yolo_out_15, yolo_out_18, yolo_out_21):
        # Clone the tensors and ensure they require gradient computation
        yolo_out_15 = yolo_out_15.clone().detach().requires_grad_(True)
        yolo_out_18 = yolo_out_18.clone().detach().requires_grad_(True)
        yolo_out_21 = yolo_out_21.clone().detach().requires_grad_(True)

        # Reduce channel dimensions
        yolo_out_15 = F.relu(self.conv1(yolo_out_15))
        yolo_out_18 = F.relu(self.conv2(yolo_out_18))
        yolo_out_21 = F.relu(self.conv3(yolo_out_21))

        # Resize feature maps to a common size
        yolo_out_15 = self.adaptive_pool(yolo_out_15)
        yolo_out_18 = self.adaptive_pool(yolo_out_18)
        yolo_out_21 = self.adaptive_pool(yolo_out_21)

        out = torch.cat((yolo_out_15, yolo_out_18, yolo_out_21), 1)
        out = F.relu(self.conv4(out))

        concatenated_features = out.flatten(1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(concatenated_features))
        # x = self.dropout1(x)  # Apply dropout after activation
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x)  # Apply dropout after activation
        x = self.fc3(x)

        return F.softmax(x, dim=1)


# implement model 2
class YOLOClassifier(nn.Module):
    def __init__(self, yolo_model, classifier_model):
        super(YOLOClassifier, self).__init__()
        self.yolo_model = yolo_model.model
        self.classifier_model = classifier_model
        self.lf = None  # low feature
        self.mf = None  # middle feature
        self.hf = None  # high feature

        self.yolo_model.model[15].register_forward_hook(self.hook_fn_low)
        self.yolo_model.model[18].register_forward_hook(self.hook_fn_middle)
        self.yolo_model.model[21].register_forward_hook(self.hook_fn_high)

    def hook_fn_low(self, module, input, output):
        self.lf = output

    def hook_fn_middle(self, module, input, output):
        self.mf = output

    def hook_fn_high(self, module, input, output):
        self.hf = output

    # Define feature extraction function
    def extract_features(self, img, layer_index=20):  # Choose the layer that fit your application
        self.yolo_model(img)
        # if self.low_features.__len__() == 1:
        return self.lf, self.mf, self.hf

    def forward(self, x):

        # yolo_out_1 = self.extract_features(x, layer_index=1) # torch.Size([1, 96, 512, 512])
        # yolo_out_2 = self.extract_features(x, layer_index=2) # torch.Size([2, 96, 512, 256])
        # yolo_out_3 = self.extract_features(x, layer_index=3) # torch.Size([2, 192, 256, 128])
        # yolo_out_4 = self.extract_features(x, layer_index=4) # torch.Size([2, 192, 256, 128])
        # yolo_out_5 = self.extract_features(x, layer_index=5) # torch.Size([2, 384, 128, 64])
        # yolo_out_6 = self.extract_features(x, layer_index=6) # torch.Size([2, 384, 128, 64])
        # yolo_out_7 = self.extract_features(x, layer_index=7) # torch.Size([2, 576, 64, 32])
        # yolo_out_8 = self.extract_features(x, layer_index=8) # torch.Size([2, 576, 64, 32])
        # yolo_out_9 = self.extract_features(x, layer_index=9) # torch.Size([2, 576, 64, 32])
        # yolo_out_10 = self.extract_features(x, layer_index=10) # torch.Size([2, 576, 128, 64])
        # yolo_out_11 = self.extract_features(x, layer_index=11) # torch.Size([2, 960, 128, 64])
        # yolo_out_12 = self.extract_features(x, layer_index=12) # torch.Size([2, 384, 128, 64])
        # yolo_out_13 = self.extract_features(x, layer_index=13) # torch.Size([2, 384, 256, 128])
        # yolo_out_14 = self.extract_features(x, layer_index=14) # torch.Size([2, 576, 256, 128])
        # yolo_out_15 = self.extract_features(x, layer_index=15) # torch.Size([2, 192, 256, 128])
        # yolo_out_16 = self.extract_features(x, layer_index=16) # torch.Size([2, 192, 128, 64])
        # yolo_out_17 = self.extract_features(x, layer_index=17) # torch.Size([2, 576, 128, 64])
        # yolo_out_18 = self.extract_features(x, layer_index=18) # torch.Size([2, 384, 128, 64])
        # yolo_out_19 = self.extract_features(x, layer_index=19) # torch.Size([2, 384, 64, 32])
        # yolo_out_20 = self.extract_features(x, layer_index=20) # torch.Size([2, 960, 64, 32])
        # yolo_out_21 = self.extract_features(x, layer_index=21) # torch.Size([2, 576, 64, 32])

        yolo_out_15, yolo_out_18, yolo_out_21 = self.extract_features(x)

        classifier_out = self.classifier_model(yolo_out_15, yolo_out_18, yolo_out_21)
        return classifier_out


# implement model 3
class ResNetClassifier(nn.Module):
    def __init__(self, resnet_model, classifier_model):
        super(ResNetClassifier, self).__init__()
        self.resnet_model = resnet_model
        self.classifier_model = classifier_model

    def forward(self, x):
        resnet_out = self.resnet_model(x)
        classifier_out = self.classifier_model(resnet_out)
        return classifier_out


@hydra.main(config_path="../../../config", config_name="config")
def main(cfg: DictConfig):

    transform = transforms.Compose(
        [
            transforms.Resize((2048, 1024)),  # Resize images
            transforms.RandomHorizontalFlip(),  # Data augmentation
            # transforms.RandomRotation(20),  # Data augmentation
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
        ]
    )

    # yolo_model = ultralytics.YOLO(
    #     "/mnt/4TB1/onj/onj_project/outputs/2024-03-12/yolo_v8m_epoch50/runs/detect/train/weights/last.pt"
    # )

    # ====================================================================================================================
    yolo_model = ultralytics.YOLO("/mnt/aix22301/onj/outputs/2024-05-06/15-20-18/runs/detect/train/weights/last.pt")

    # Freeze YOLO model
    # for param in yolo_model.parameters():
    #     param.requires_grad = False

    classifier_model = ClassifierModel(num_classes=2)

    model = YOLOClassifier(yolo_model, classifier_model)
    # ====================================================================================================================
    # yolo_model = ultralytics.YOLO("/mnt/aix22301/onj/outputs/2024-05-06/15-20-18/runs/detect/train/weights/last.pt")

    dataset = ImageFolder(root="/mnt/aix22301/onj/dataset/v0/CLS_PA/", transform=transform)

    # Splitting dataset into train and validation
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_PA.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.train_PA.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()  # Combines a Sigmoid layer and the BCELoss in one single class

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_PA.lr, weight_decay=cfg.train_PA.weight_decay)

    # check if cuda is available
    print("Checking GPU availability: ", torch.cuda.is_available())
    print("Number of available gpu counts: ", torch.cuda.device_count())

    device = f"cuda:{cfg.train.gpu}"

    # train in 10 epochs with gpu
    model.to(device)
    best_f1_score = 0.0

    for epoch in range(cfg.train_PA.epoch):
        wandb.log({"lr": cfg.train_PA.lr})
        model.yolo_model.model.train()
        model.classifier_model.train()

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        cumulated_loss = 0.0

        for inputs, labels in tqdm(train_loader):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            cumulated_loss += loss.item()
            optimizer.step()

            outputs = torch.argmax(outputs, dim=1)
            for i in range(len(outputs)):
                if outputs[i] == 1 and labels[i] == 1:
                    TP += 1
                elif outputs[i] == 1 and labels[i] == 0:
                    FP += 1
                elif outputs[i] == 0 and labels[i] == 0:
                    TN += 1
                elif outputs[i] == 0 and labels[i] == 1:
                    FN += 1

        print(
            f"[TRAIN] Epoch {epoch+1}/{cfg.train_PA.epoch}, Training Loss: {cumulated_loss/(len(train_loader)*2):3f}, Accuracy: {(TP + TN) / (TP + TN + FP + FN):3f}, Precision: {TP / (TP + FP):3f}, Recall: {TP / (TP + FN):3f}, F1 Score: {2 * (TP / (TP + FP)) * (TP / (TP + FN)) / (TP / (TP + FP) + TP / (TP + FN)):3f}"
        )
        wandb.log(
            {
                "train_loss": cumulated_loss / (len(train_loader) * 2),
                "train_accuracy": (TP + TN) / (TP + TN + FP + FN),
                "train_precision": TP / (TP + FP),
                "train_recall": TP / (TP + FN),
                "train_f1_score": 2 * (TP / (TP + FP)) * (TP / (TP + FN)) / (TP / (TP + FP) + TP / (TP + FN)),
            }
        )

        model.yolo_model.model.eval()
        model.classifier_model.eval()
        with torch.no_grad():
            TP = 0
            FP = 0
            TN = 0
            FN = 0

            cumulated_loss = 0.0
            for inputs, labels in tqdm(validation_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                cumulated_loss += loss.item()

                outputs = torch.argmax(outputs, dim=1)
                for i in range(len(outputs)):
                    if outputs[i] == 1 and labels[i] == 1:
                        TP += 1
                    elif outputs[i] == 1 and labels[i] == 0:
                        FP += 1
                    elif outputs[i] == 0 and labels[i] == 0:
                        TN += 1
                    elif outputs[i] == 0 and labels[i] == 1:
                        FN += 1
            try:
                print(
                    f"[VALID] Epoch {epoch+1}/{cfg.train_PA.epoch}, Validation Loss: {cumulated_loss/(len(validation_loader)*2):3f}, Accuracy: {(TP + TN) / (TP + TN + FP + FN):3f}, Precision: {TP / (TP + FP):3f}, Recall: {TP / (TP + FN):3f}, F1 Score: {2 * (TP / (TP + FP)) * (TP / (TP + FN)) / (TP / (TP + FP) + TP / (TP + FN)):3f}"
                )
                f1_score = 2 * (TP / (TP + FP)) * (TP / (TP + FN)) / (TP / (TP + FP) + TP / (TP + FN))
                wandb.log(
                    {
                        "val_loss": cumulated_loss / (len(train_loader) * 2),
                        "val_accuracy": (TP + TN) / (TP + TN + FP + FN),
                        "val_precision": TP / (TP + FP),
                        "val_recall": TP / (TP + FN),
                        "val_f1_score": f1_score,
                    }
                )
                if best_f1_score < f1_score:
                    best_f1_score = f1_score
                    torch.save(model.state_dict(), f"backbone_PA_best_model.pth")
                    print(f"Best model saved with F1 score: {best_f1_score}")

            except:
                pass


if __name__ == "__main__":
    wandb.init(
        project=f"ONJ_classification",
        name="PA_backbone",
    )
    main()
