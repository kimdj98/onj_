# Standard library imports
import os

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# DEBUG: Check gradients
def check_gradients(named_parameters):
    for name, param in named_parameters:
        if param.requires_grad:
            if param.grad is None:
                print(f"Parameter {name} has no gradient.")
            else:
                print(f"Parameter {name} gradient: {param.grad.abs().mean()}")


def save_file(img_tensor, filename="preview_img.jpg"):
    """
    Save the image tensor to a file.
    
    Args:
        img_tensor (torch.Tensor): The image tensor to save.
        filename (str): The filename to save the image as.
    """
    img_np = img_tensor.cpu().float().numpy()
    img_np = np.transpose(img_np[0], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    plt.imsave(filename, img_np)
    

def get_cls_pa_attention_map(attn_weights, img, patch_size, B=1):
    """
    Get the attention map for the CLS token and PA tokens.
    Dynamically adapts to input image dimensions and patch size.
    
    Args:
        attn_weights: attention weights from transformer
        img: input image tensor of shape (B, C, H, W)
        patch_size: tuple of (H, W) for patch dimensions
        B: batch size
    Returns:
        Attention map tensor of same spatial dimensions as input image
    """
    _, _, H, W = img.shape
    grid_size = [H // patch_size[0], W // patch_size[1]]
    num_patches = grid_size[0] * grid_size[1]

    try:
        # Extract CLS token's attention to PA tokens
        cls_pa_attn = attn_weights[:, :, 0, -num_patches:]  # (B, num_heads, num_patches)
        cls_pa_attn_mean = cls_pa_attn.mean(dim=1)  # Average over heads

        # Reshape to grid size and upsample to original image size
        attn_map = cls_pa_attn_mean.reshape(B, grid_size[0], grid_size[1])
        attn_map = F.interpolate(attn_map.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False)
        attn_map = attn_map.squeeze(1)

        # Normalize attention map
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map
        
    except Exception as e:
        print(f"Warning: Error in attention map generation: {e}")
        print(f"Image shape: {img.shape}, Patch size: {patch_size}, Grid size: {grid_size}")
        print(f"Attention weights shape: {attn_weights.shape}, Num patches: {num_patches}")
        # Return a default attention map in case of error
        return torch.ones((B, H, W), device=img.device)

def visualize_cls_pa_attention_map(attn_weights, img, model_config, save_path="attention_map.png"):
    """
    Visualize the attention map for the CLS token and PA tokens.
    Automatically adapts to model configuration and input dimensions.
    
    Args:
        attn_weights: attention weights from transformer
        img: input image tensor
        model_config: Config object containing model configuration
        save_path: path to save the visualization
    """
    # Get attention map using current model's patch size
    attn_map = get_cls_pa_attention_map(attn_weights, img, model_config.n_patch2d)

    # Calculate figure size based on image aspect ratio
    _, _, H, W = img.shape
    aspect_ratio = W / H
    fig_height = 5
    fig_width = fig_height * aspect_ratio

    # Prepare image for visualization
    img_np = img.cpu().float().numpy()[0].transpose(1, 2, 0)
    attn_map_np = attn_map.cpu().numpy()[0]

    # Create figure with proper aspect ratio
    plt.figure(figsize=(fig_width * 2.2, fig_height))  # Wider figure for better subplot layout
    
    # Create subplots for better visualization
    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(attn_map_np, cmap='jet')
    plt.title("Attention Map")
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_np, cmap='gray')
    plt.imshow(attn_map_np, cmap='jet', alpha=0.5)
    plt.title(f"Overlay (patch: {model_config.n_patch2d}, img: {H}x{W})")
    plt.colorbar()
    plt.axis('off')
    
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Adjust layout and save with high quality
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    

class Logger:
    """
    A logger class that handles logging training progress, script contents, and dataset information.
    """
    def __init__(self, log_file, dataset_yaml):
        self.log_file = log_file
        self.log_script_file = log_file.replace("log.txt", "log_script.py")
        self.log_data_file = log_file.replace("log.txt", "log_data.txt")
        self.dataset_yaml = dataset_yaml
        
        # Initialize log files
        for file in [self.log_file, self.log_script_file, self.log_data_file]:
            with open(file, "w") as f:
                pass

        self.log_script()
        self.log_data()
        self.step = 0

    def log(self, message):
        self.step += 1
        with open(self.log_file, "a") as f:
            f.write(f"{self.step} " + message)

    def log_script(self):
        # Open current file and log every lines of code inside the file
        import inspect
        current_file = inspect.stack()[1].filename
        with open(current_file, "r") as f:
            lines = f.readlines()

        with open(self.log_script_file, "a") as f2:
            f2.writelines(lines)

    def log_data(self):
        with open(self.dataset_yaml, "r") as f:
            lines = f.readlines()

        with open(self.log_data_file, "a") as f:
            f.writelines(lines)

    def resume(self, resume_file):
        # Open the existing log file to read its content
        with open("/".join(resume_file.split("/")[:-1]) + "/log.txt", "r") as f:
            lines = f.readlines()  # Read all lines

        # Write the content to the new log file
        with open(self.log_file, "w") as f2:
            f2.writelines(lines)

        # Get the last step number from the lines read
        if lines:
            self.step = int(lines[-1].split(" ")[0])
    
