import ultralytics
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.1", config_path="../../../config", config_name="config")
def main(cfg: DictConfig):
    model = ultralytics.YOLO("yolov8n.pt")

    dataset_yaml = "/mnt/4TB1/onj/onj_project/data/yolo_dataset.yaml"

    model.train(data=dataset_yaml, epochs=300, device="3", imgsz=512, scale=0.0, mosaic=1.0)

    results = model.predict(
        "/mnt/4TB1/onj/dataset/v0/YOLO/images/train/EW-0012_20200216_MDCT_axial_60.jpg",
        save=True,
    )


if __name__ == "__main__":
    main()
