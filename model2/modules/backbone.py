import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics
from ultralytics.nn.modules.head import Classify, Detect  # add classify to layer number 8
from ultralytics.utils.loss import v8DetectionLoss, v8ClassificationLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

import hydra
from omegaconf import DictConfig

import sys

sys.path.append("/mnt/aix22301/onj/code/")
from model2.modules.utils import preprocess_data

# def get_ct_backbone():
#     # Implement your custom ct_backbone here
#     pass


# def get_pa_backbone():
#     pa_model = ultralytics.YOLO("yolov8n.pt")
#     pa_model = pa_model.model
#     return pa_model


# def get_classifier():
#     # Classifier with input channels 1 and 2 output classes
#     return Classify(c1=3, c2=2)


# if __name__ == "__main__":
#     pa_backbone = get_pa_backbone()
#     ct_backbone = get_ct_backbone()
#     classifier = get_classifier()

#     yolo_model = pa_backbone
#     input_tensor = torch.randn(4, 3, 224, 224)
dataset_yaml = "/mnt/aix22301/onj/code/data/yolo_dataset.yaml"

version = "yolov8n"
#     model = ultralytics.YOLO(f"{version}.pt")

#     # Modify the dataset yaml to have 2 classes
args = {
    "task": "detect",
    "data": dataset_yaml,
    "imgsz": 640,
    "single_cls": False,
    "model": f"{version}.pt",
    "mode": "train",
}


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


@hydra.main(version_base="1.1", config_path="../../config", config_name="config")
def main(cfg):
    base_path = cfg.data.data_dir
    # from ultralytics.data import build_dataloader, build_yolo_dataset
    # from ultralytics.utils import check_dataset

    # data = check_dataset("path/to/data.yaml")  # verify dataset integrity

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

    version = "yolov8n.pt"
    # model = ultralytics.YOLO(f"{version}.pt")

    # Modify the dataset yaml to have 2 classes
    args = {
        "task": "detect",
        "data": custom_yaml,
        "imgsz": 640,
        "single_cls": False,
        "model": f"{version}",
        "mode": "train",
        "stride": 1,
    }

    args = {
        "model": "/mnt/aix22301/onj/code/data/yolo_dataset.yaml",
        "imgsz": [2048, 1024],
        "task": "detect",
        "data": "/mnt/aix22301/onj/code/data/yolo_dataset.yaml",
        "mode": "train",
    }

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

    pbar = enumerate(train_loader)
    nb = len(train_loader)
    nw = max(round(trainer.args.warmup_epochs * nb), 100) if trainer.args.warmup_epochs > 0 else -1  # warmup iterations

    pbar = TQDM(enumerate(train_dataset), total=nb)
    for i, data in pbar:
        with torch.cuda.amp.autocast(trainer.amp):
            data = trainer.preprocess_batch(data)
            data = preprocess_data(base_path, data)  # adds CT data and unify the data device
            data["img"] = data["img"].unsqueeze(0)
            data["CT_image"] = data["CT_image"].unsqueeze(0)

            trainer.loss, trainer.loss_items = trainer.model.loss(data)
            trainer.tloss = (
                (trainer.tloss * i + trainer.loss_items) / (i + 1) if trainer.tloss is not None else trainer.loss_items
            )

            # Backward
            trainer.scaler.scale(trainer.loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            trainer.optimizer_step()  # includes zero_grad()

    trainer.train(data=dataset_yaml)
    # yolo_model.train()

    yolo_model = YOLO("yolov8n.pt")
    yolo_model.train(data=dataset_yaml)
    yolo_model.train_loader = yolo_model.get_dataloader()

    # overrides = yaml_load(checks.check_yaml(args)) if args else overrides
    yolo_model
    model = yolo_model.model

    model2 = YOLO(model="yolov8n.pt").model

    model(input_tensor)
    model2(input_tensor)

    trainer = DetectionTrainer(overrides=args)
    # dataloader = trainer.get_dataloader(dataset_yaml, batch_size=16, rank=0, mode="train")
    # trainer.get_model()
    # Create an instance of the YOLO model
    # model_path = "yolov8n.pt"
    # yolo_model = YOLO(model_path)
    yolo_model = yolo_model.model
    yolo_model = DetectionModel(cfg=None, nc=2, verbose=True)
    yolo_model.args = args
    detection_loss_fn = v8DetectionLoss(yolo_model)

    # yolo_model = yolo_model.model

    # Register the hook to capture the 8th layer feature map
    register_hook(yolo_model, 8)

    # Example input tensor
    input_tensor = torch.randn(4, 3, 224, 224)

    # Perform a forward pass through the model
    _ = yolo_model(input_tensor)

    # Now the feature_map variable contains the output of the 8th layer
    print("Feature map shape at layer 8:", feature_map.shape)


if __name__ == "__main__":
    main()


#     # trainer = model._smart_load("trainer")(args, model.callbacks)

#     # Example of how to train the model with the provided dataset
#     # model.train(data=dataset_yaml)

#     # Perform a forward pass through the detection model and classifier
#     detector_output = yolo_model(input_tensor)
#     classifier_output = classifier(input_tensor)

#     print("Detector Output shape:", detector_output[0].shape)
#     print("Classifier Output shape:", classifier_output.shape)


# import torch
# import torch.nn as nn
# from ultralytics import YOLO


# class YOLOWithFeatureExtraction(YOLO):
#     def __init__(self, model_path):
#         super(YOLOWithFeatureExtraction, self).__init__(model_path)
#         self.model = self.model.model  # Accessing the internal model from YOLO

#     def forward(self, x):
#         features = None
#         for i, layer in enumerate(self.model):
#             x = layer(x)
#             if i == 8:
#                 features = x  # Capture the feature map at layer 8
#         return features, x  # Return the feature map and final output


# # # Create an instance of the modified model
# # model_path = "yolov8n.pt"
# # yolo_model = YOLOWithFeatureExtraction(model_path)

# # # Example input tensor
# # input_tensor = torch.randn(4, 3, 224, 224)

# # # Forward pass through the model to get the feature map and final output
# # features, final_output = yolo_model(input_tensor)

# # print("Feature map shape at layer 8:", features.shape)
# # print("Final output shape:", final_output.shape)


# if __name__ == "__main__":
#     model_path = "yolov8n.pt"

#     # Create the model instance
#     yolo_model = YOLOWithFeatureExtraction(model_path)

#     # Example input tensor
#     input_tensor = torch.randn(4, 3, 224, 224)

#     # Perform a forward pass to get the feature map at layer 8 and the final output
#     features, final_output = yolo_model(input_tensor)

#     print("Feature map shape at layer 8:", features.shape)
#     print("Final output shape:", final_output.shape)
