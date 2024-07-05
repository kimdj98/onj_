# # scripts/train.py
# import sys
# sys.path.append("/mnt/aix22301/onj/code")

# import time
# from pathlib import Path
# import torch
# import hydra
# import wandb
# from monai.data import Dataset, DataLoader
# from data.data_dicts import get_data_dicts
# from model.backbone.classifier.ResNet3D import resnet18_3d, resnet50_3d
# from model.backbone.classifier.backbone_PA_2D import ClassifierModel, YOLOClassifier
# import ultralytics
# from model.fusor.concat import ConcatModel
# from model.backbone.utils import FeatureExpand
# from utils.metrics import get_metrics
# from utils.plotting import plot_auroc
# from utils.training import train_one_epoch, validate

# @hydra.main(version_base="1.1", config_path="../config", config_name="config")
# def train(cfg):
#     wandb.init(project="ONJ_classification", name=f"{cfg.train.description}")

#     # Load Data
#     transforms = get_transforms(cfg)
#     BASE_PATH = Path(cfg.data.data_dir)
#     train_data_dicts, val_data_dicts, test_data_dicts = get_data_dicts(
#         BASE_PATH, includes=[Modal.CBCT, Modal.MDCT, Modal.PA], random_state=cfg.data.random_state
#     )

#     train_dataset = Dataset(data=train_data_dicts, transform=transforms)
#     train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)

#     val_dataset = Dataset(data=val_data_dicts, transform=transforms)
#     val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=True)

#     test_dataset = Dataset(data=test_data_dicts, transform=transforms)
#     test_dataloader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=True)

#     # Initialize Models
#     model_3d = resnet18_3d() if cfg.model.CT == "resnet18" else resnet50_3d()
#     if cfg.train.pretrained_CT:
#         model_3d.load_state_dict(torch.load(cfg.train.pretrained_CT))

#     yolo_model = ultralytics.YOLO(cfg.model.PA)
#     classifier_model = ClassifierModel(num_classes=2)
#     model_2d = YOLOClassifier(yolo_model, classifier_model)
#     if cfg.train.pretrained_PA:
#         model_2d.load_state_dict(torch.load(cfg.train.pretrained_PA))

#     model = ConcatModel(model_2d, model_3d)
#     if cfg.train.pretrained_fusion:
#         model.load_state_dict(torch.load(cfg.train.pretrained_fusion))

#     device = torch.device(f"cuda:{cfg.train.gpu}" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     feature_expand_low = FeatureExpand(in_channels=cfg.model.low_channels, out_channels=cfg.model.low_expanded).to(device)
#     feature_expand_middle = FeatureExpand(in_channels=cfg.model.mid_channels, out_channels=cfg.model.mid_expanded).to(device)
#     feature_expand_high = FeatureExpand(in_channels=cfg.model.high_channels, out_channels=cfg.model.high_expanded).to(device)

#     loss_fn = CrossEntropyLoss()
#     auroc, acc = get_metrics()
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**)
