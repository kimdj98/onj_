# utils/plotting.py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_auroc(y_true, y_scores, epoch, title="roc"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - Epoch {epoch}")
    plt.legend(loc="lower right")
    plt.savefig(f"{title}.png")
    plt.close()


import torch
import numpy as np
import torch.nn as nn
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def apply_grad_cam(model: nn.Module, target_layer, input_tensor, grad_cam=GradCAMPlusPlus):
    # with torch.no_grad():
    model.eval()
    # Assume the model and target_layer are correctly configured and passed
    cam = grad_cam(model=model, target_layers=target_layer)
    target_category = 1  # You can specify the category index if needed

    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.zero_()

    # You can create a custom forward function for your model to return the output logits
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category)])
    grayscale_cam = (grayscale_cam - np.min(grayscale_cam)) / (np.max(grayscale_cam) - np.min(grayscale_cam))

    # In your case, you might have preprocessed images or raw images
    # Convert your input tensor to image (Numpy array) format as needed
    image = np.moveaxis(input_tensor["PA_image"].cpu().numpy(), 1, -1)  # Adjust depending on your preprocessing
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0,1] for visualization

    visualization = show_cam_on_image(image, grayscale_cam[0, :], use_rgb=True)
    return visualization


def apply_grad_cam_3d(model, target_layer, input_tensor, slice_idx):
    # Assume the model and target_layer are correctly configured and passed
    cam = GradCAM(model=model, target_layers=target_layer)
    target_category = 1  # Assuming binary classification for simplicity

    # Compute the CAM
    grayscale_cams = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category)])

    # Select the CAM for the desired slice
    cam_slice = grayscale_cams[0][slice_idx, :, :]  # Adjust indexing based on your CAM tensor shape

    # Normalize the CAM slice for better visualization
    cam_slice = (cam_slice - cam_slice.min()) / (cam_slice.max() - cam_slice.min())

    # Assuming you have a corresponding CT slice ready for visualization
    ct_slice = input_tensor["CT_image"][0, 0, :, :, slice_idx].cpu().numpy()  # 0 for channel if grayscale
    ct_slice = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())

    # Apply the CAM mask to the CT image slice
    visualization = show_cam_on_image(ct_slice, cam_slice, use_rgb=False)
    return visualization


from medcam import medcam
from medcam import medcam_inject
import os


def apply_med_cam(model, target_layer, input):
    model.eval()
    os.makedirs("./attention_maps", exist_ok=True)
    model = medcam_inject.inject(
        model,
        output_dir="./attention_maps",
        backend="gcam",
        layer="auto",  # auto: Selects the last layer from which attention maps can be extracted.
        # label=1,
        save_maps=True,
    )
    input.permute(0, 1, 4, 2, 3)
    output = model(input)


# # Example of using the function
# model.eval()  # Make sure the model is in evaluation mode
# input_tensor = ...  # This should be your preprocessed input for which you want the CAM
# target_layer = model.features  # Adjust this to match your model's appropriate layer

# cam_image = apply_grad_cam(model, target_layer, input_tensor)
# plt.imshow(cam_image)
# plt.show()
