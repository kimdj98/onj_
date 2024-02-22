# this script is used to distribute the metadata to each patient folder.
# The metadata is stored in a JSON file and contains the bounding box coordinates for each patient.
import os
import json
import yaml
from pathlib import Path
from tqdm import tqdm

curr_dir = Path(os.getcwd())
print("Current Dir:", curr_dir)  # Current Dir: path to ONJ folder

base_dir = curr_dir / "dataset" / "v0"
print("Base Dir:", base_dir)  # Base Dir: path to ONJ folder/{version}

CTS = ["MDCT", "CBCT"]
DIRECTIONS = ["axial", "coronal"]

# Step 1: Read the JSON file
with open("./dataset/v0/label_v231122/label.json", "r") as json_file:
    json_data = json.load(json_file)

    # Step 2: write the JSON data to each patient folder
    for patient in tqdm(json_data["patients"]):
        patient_name = patient["patient_name"]
        patient_dir = base_dir / "ONJ" / patient_name

        has_CT = False
        for CT in CTS:  # check if the patient has CT data
            if CT in patient.keys():
                has_CT = True
                modal = CT
                break

        if not has_CT:  # if the patient does not have CT data, skip
            print(f"Patient {patient_name} does not have CT data")
            continue

        if not patient_dir.exists():  # if the patient does not exist, skip
            print(f"Patient {patient_name} does not exist")
            continue

        # if (patient_dir / "label.json").exists(): # if the patient already has label.json, skip
        #     print(f"Patient {patient_name} already has label.json")
        #     continue

        with open(patient_dir / "label.json", "w") as patient_json_file:
            json.dump(patient, patient_json_file)

        # Step 3: distribute the metadata to each direction
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
                slice["bbox"][0]["coordinates"][0] /= width
                slice["bbox"][0]["coordinates"][1] /= height
                slice["bbox"][0]["coordinates"][2] /= width
                slice["bbox"][0]["coordinates"][3] /= height

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

# NOTE: EW-0070 does not exist.
# NOTE: EW-0141 does not exist.
