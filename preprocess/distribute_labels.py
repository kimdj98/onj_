# this script is used to distribute the metadata to each patient folder.
# The metadata is stored in a JSON file and contains the bounding box coordinates for each patient.
import os
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    base_dir = Path(cfg.data.data_dir)
    print("Base Dir:", base_dir)  # Base Dir: path to ONJ folder/{version}

    CTS = ["MDCT", "CBCT"]
    DIRECTIONS = ["axial", "coronal"]

    # Step 1: Read the JSON file
    with open(cfg.data.label_dir, "r") as json_file:
        json_data = json.load(json_file)

        # Step 2: write the JSON data to each patient folder
        for patient in tqdm(json_data["patients"]):
            patient_name = patient["patient_name"]
            patient_dir = base_dir / "ONJ_labeling" / patient_name

            # check if the patient_label has CT data
            has_CT = False
            for CT in CTS:
                if CT in patient.keys():
                    has_CT = True
                    modal = CT
                    break

            if not has_CT:  # if the patient does not have CT data, skip
                print(f"Patient {patient_name} does not have CT label")
                continue

            if not patient_dir.exists():  # if the patient does not exist, skip
                print(f"Patient {patient_name} does not exist")
                continue

            if not (patient_dir / modal).exists():  # if the patient does not have CT data, skip
                print(f"Patient {patient_name} does not have {modal} data")
                continue

            with open(patient_dir / "label.json", "w") as patient_json_file:
                json.dump(patient, patient_json_file)

            # Step 3: distribute the metadata to each direction
            folders = list((patient_dir / modal).glob("*"))
            if folders == []:
                print(f"Patient {patient_name} does not have {modal} data")
                continue
            CT_with_date = list((patient_dir / modal).glob("*"))[0]
            modal_metadata = patient[modal]

            for dir in DIRECTIONS:
                CT_dir = CT_with_date / (modal + "_" + dir)
                assert CT_dir.exists(), f"{CT_dir} does not exist."
                modal_dir_metadata = modal_metadata[dir]

                width = modal_dir_metadata["width"]
                height = modal_dir_metadata["height"]

                # normalize the bounding box coordinates
                for slice in modal_dir_metadata["slices"]:
                    x = slice["bbox"][0]["coordinates"][0]
                    y = slice["bbox"][0]["coordinates"][1]
                    w = slice["bbox"][0]["coordinates"][2]
                    h = slice["bbox"][0]["coordinates"][3]

                    slice["bbox"][0]["coordinates"][0] = (x + w / 2) / width
                    slice["bbox"][0]["coordinates"][1] = (y + h / 2) / height
                    slice["bbox"][0]["coordinates"][2] = w / width
                    slice["bbox"][0]["coordinates"][3] = h / height

                # check if the bounding box coordinates are well normalized
                assert (
                    slice["bbox"][0]["coordinates"][0] <= 1 and slice["bbox"][0]["coordinates"][0] >= 0
                ), f"Invalid x coordinate: {slice['bbox'][0]['coordinates'][0]}"
                assert (
                    slice["bbox"][0]["coordinates"][1] <= 1 and slice["bbox"][0]["coordinates"][1] >= 0
                ), f"Invalid y coordinate: {slice['bbox'][0]['coordinates'][1]}"
                assert (
                    slice["bbox"][0]["coordinates"][2] <= 1 and slice["bbox"][0]["coordinates"][2] >= 0
                ), f"Invalid width: {slice['bbox'][0]['coordinates'][2]}"
                assert (
                    slice["bbox"][0]["coordinates"][3] <= 1 and slice["bbox"][0]["coordinates"][3] >= 0
                ), f"Invalid height: {slice['bbox'][0]['coordinates'][3]}"

                # write the metadata to the direction folder
                with open(CT_dir / "label.json", "w") as CT_json_file:
                    json.dump(modal_dir_metadata, CT_json_file)


if __name__ == "__main__":
    main()

# NOTE: EW-0070 does not CT data.
# NOTE: EW-0141 does not CT data.
# NOTE: EW-0424 does not have CBCT data but has CT label
