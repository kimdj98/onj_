import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import json
import shutil


def create_folder_structure(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_path, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_path, "labels", split), exist_ok=True)


def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def find_valley_points(image):
    row_mean = np.mean(image, axis=1)
    max_index_upper = np.argmax(row_mean[: image.shape[0] // 2])
    min_index_upper = np.argmin(row_mean[max_index_upper : image.shape[0] // 2]) + max_index_upper
    max_index_lower = np.argmax(row_mean[image.shape[0] // 2 :]) + image.shape[0] // 2
    min_index_lower = np.argmin(row_mean[max_index_lower:]) + max_index_lower
    return min_index_upper, min_index_lower


def crop_vertical(image, min_index_upper, min_index_lower, adjust=150, constraints=None):
    if constraints:
        min_index_upper = min(min_index_upper, constraints["y_min"])
        min_index_lower = max(min_index_lower, constraints["y_max"])
    if min_index_upper - adjust < 0:
        min_index_upper = 0
        cropped_image = image[:min_index_lower, :]
    else:
        min_index_upper = min_index_upper - adjust
        cropped_image = image[min_index_upper:min_index_lower, :]
    return cropped_image, min_index_upper, min_index_lower


def crop_columns_pre(image, constraints=None):
    col_mean = np.mean(image, axis=0)
    min_index_left = np.argmin(col_mean[: image.shape[1] // 12])
    min_index_right = np.argmin(col_mean[-image.shape[1] // 12 :]) + image.shape[1] - image.shape[1] // 12
    if constraints:
        min_index_left = min(min_index_left, constraints["x_min"])
        min_index_right = max(min_index_right, constraints["x_max"])
    cropped_image = image[:, min_index_left:min_index_right]
    return cropped_image, min_index_left, min_index_right


def crop_columns(image, Q=35, constraints=None):
    cols_to_check = image.shape[1] // 6
    col_mean = np.mean(image[:, -cols_to_check:], axis=0)
    min_col_index = np.argmin(col_mean)
    cropped_image = image[:, : -(cols_to_check - min_col_index)]

    col_mean = np.mean(image[:, :cols_to_check], axis=0)
    min_col_index = np.argmin(col_mean)
    cropped_image = cropped_image[:, min_col_index:]

    theta = (55 / 16) * Q
    _, thresholded_image = cv2.threshold(cropped_image, theta, 255, cv2.THRESH_BINARY)
    blocks = 200
    block_size = (cropped_image.shape[0] // blocks, cropped_image.shape[1] // blocks)
    homogeneous_image = np.zeros_like(thresholded_image)

    for i in range(blocks):
        for j in range(blocks):
            block = thresholded_image[
                i * block_size[0] : (i + 1) * block_size[0], j * block_size[1] : (j + 1) * block_size[1]
            ]
            variance = np.var(block)
            if variance > theta:
                homogeneous_image[
                    i * block_size[0] : (i + 1) * block_size[0], j * block_size[1] : (j + 1) * block_size[1]
                ] = 255

    labeled_image = label(homogeneous_image)

    for region in regionprops(labeled_image):
        if region.area < 50:
            for coordinates in region.coords:
                labeled_image[coordinates[0], coordinates[1]] = 0

    column_sum = np.sum(labeled_image, axis=0)
    left_bound = 200
    right_bound = labeled_image.shape[1] - 200

    left_crop_index = np.where(column_sum[left_bound:] > 0)[0][0] + left_bound
    right_crop_index = np.where(column_sum[:right_bound] > 0)[-1][-1]

    if constraints:
        left_crop_index = min(left_crop_index, constraints["x_min"])
        right_crop_index = max(right_crop_index, constraints["x_max"])

    final_cropped_image = cropped_image[:, left_crop_index:right_crop_index]
    return final_cropped_image, left_crop_index + min_col_index, right_crop_index

    return cropped_image, min_col_index, cropped_image.shape[1]


def process(image_path, output_image_path, output_label_path, bbox_label_path):
    image = read_image(image_path)

    min_x = image.shape[1]
    max_x = 0
    min_y = image.shape[0]
    max_y = 0

    if bbox_label_path is None:
        constraints = None
    else:
        with open(bbox_label_path, "r") as f:
            annotations = json.load(f)
            try:
                bboxes = annotations["bbox"]
                for bbox in bboxes:
                    x, y, w, h = bbox["coordinates"]
                    x1 = int((x - w / 2) * image.shape[1])
                    x2 = int((x + w / 2) * image.shape[1])
                    min_x = min(x1, min_x)
                    max_x = max(x2, max_x)
                    y1 = int((y - h / 2) * image.shape[0])
                    y2 = int((y + h / 2) * image.shape[0])
                    min_y = min(y1, min_y)
                    max_y = max(y2, max_y)
            except:  # if there is no bbox (for non-onj data)
                min_x = 0
                max_x = 1 * image.shape[1]
                min_y = 0
                max_y = 1 * image.shape[0]

        constraints = {"x_min": min_x, "x_max": max_x, "y_min": min_y, "y_max": max_y}

    image, min_index_left_pre, max_index_right_pre = crop_columns_pre(image, constraints=constraints)

    min_index_upper, min_index_lower = find_valley_points(image)

    vertically_cropped_image, min_index_upper, min_index_lower = crop_vertical(
        image, min_index_upper, min_index_lower, constraints=constraints
    )

    final_cropped_image, min_index_left, max_index_right = crop_columns(
        vertically_cropped_image, constraints=constraints
    )

    if bbox_label_path:
        try:
            for bbox in bboxes:
                x, y, w, h = bbox["coordinates"]
                bbox["coordinates"] = [
                    (x * annotations["width"] - (min_index_left_pre + min_index_left)) / final_cropped_image.shape[1],
                    (y * annotations["height"] - min_index_upper) / final_cropped_image.shape[0],
                    w * annotations["width"] / final_cropped_image.shape[1],
                    h * annotations["height"] / final_cropped_image.shape[0],
                ]

        except:
            pass

        with open(output_label_path, "w") as f:
            json.dump(annotations, f, indent=4)

    plt.imsave(output_image_path, final_cropped_image, cmap="gray")


def main():
    source_base = "/mnt/aix22301/onj/dataset/v0/YOLO_PA2"
    dest_base = "/mnt/aix22301/onj/dataset/v0/YOLO_PA2_cropped"

    create_folder_structure(dest_base)

    for split in ["train", "val", "test"]:
        image_dir = os.path.join(source_base, "images", split)
        label_dir = os.path.join(source_base, "labels", split)

        for image_name in os.listdir(image_dir):
            if image_name.endswith(".jpg"):
                image_path = os.path.join(image_dir, image_name)
                label_path = os.path.join(label_dir, image_name.replace(".jpg", ".json"))

                output_image_path = os.path.join(dest_base, "images", split, image_name)
                output_label_path = os.path.join(dest_base, "labels", split, image_name.replace(".jpg", ".json"))

                process(image_path, output_image_path, output_label_path, bbox_label_path=label_path)


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     bbox_path = None
#     # Example usage
#     # image_path = "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0028/panorama/20160515.jpg"
#     # bbox_path = "/mnt/aix22301/onj/dataset/v0/YOLO_PA/labels/train/EW-0028.json"
#     # image_path = "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0012/panorama/20200213.jpg"
#     # bbox_path = "/mnt/aix22301/onj/dataset/v0/YOLO_PA/labels/train/EW-0012.json"
#     # image_path = "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0009/panorama/20200928.jpg"
#     # bbox_path = "/mnt/aix22301/onj/dataset/v0/YOLO_PA/labels/train/EW-0009.json"
#     # image_path = "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0001/panorama/20190724.jpg"
#     # bbox_path = "/mnt/aix22301/onj/dataset/v0/YOLO_PA/labels/train/EW-0001.json"
#     # image_path = "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0013/panorama/20210408.jpg"
#     # bbox_path = "/mnt/aix22301/onj/dataset/v0/YOLO_PA/labels/train/EW-0013.json"
#     # image_path = "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0008/panorama/20200506.jpg"
#     # bbox_path = "/mnt/aix22301/onj/dataset/v0/YOLO_PA/labels/val/EW-0008.json"
#     # image_path = "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0019/panorama/20210506.jpg"
#     # bbox_path = "/mnt/aix22301/onj/dataset/v0/YOLO_PA/labels/test/EW-0019.json"
#     # image_path = "/mnt/aix22301/onj/dataset/v0/YOLO_PA/images/test/EW-0003.jpg"
#     # bbox_path = "/mnt/aix22301/onj/dataset/v0/YOLO_PA/labels/test/EW-0003.json"
#     image_path = "/mnt/aix22301/onj/dataset/v0/YOLO_PA/images/test/EW-0019.jpg"
#     bbox_path = "/mnt/aix22301/onj/dataset/v0/YOLO_PA/labels/test/EW-0019.json"
#     output_path_original = "original_image.jpg"
#     output_path_cropped = "cropped_image.jpg"

#     # ========================  ========================
#     # Process the image in YOLO_CLS_PA into YOLO_CLS_PA_cropped

#     for folder in YOLO_CLS_PA_folders:
#         image_path = f"/mnt/aix22301/onj/dataset/v0/YOLO_CLS_PA/images/{folder}.jpg"
#         bbox_path = f"/mnt/aix22301/onj/dataset/v0/YOLO_CLS_PA/labels/{folder}.json"
#         output_path_original = f"/mnt/aix22301/onj/dataset/v0/YOLO_CLS_PA_cropped/images/{folder}.jpg"
#         output_path_cropped = f"/mnt/aix22301/onj/dataset/v0/YOLO_CLS_PA_cropped/images/{folder}.jpg"

#     process(image_path, output_path_original, output_path_cropped, bbox_label_path=bbox_path)

#     # move YOLO_PA images to YOLO_PA_cropped and change the bounding box coordinates
