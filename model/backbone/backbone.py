import hyddra
from dictconfig import DictConfig
from detector import get_YOLOv8


@hydra(main_config_path="../config", main_config_name="config")
def get_backbone(cfg: DictConfig):
    if cfg.backbone == "YOLOv8":
        return

    else:
        raise NotImplementedError(f"Backbone {cfg.backbone} not implemented")
