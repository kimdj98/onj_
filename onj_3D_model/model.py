"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""

from torch import nn
from torchsummary import summary
import torch
import time



class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    # def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
    #     super(Conv3DBlock, self).__init__()
    #     self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
    #     self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
    #     self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
    #     self.bn2 = nn.BatchNorm3d(num_features=out_channels)
    #     self.relu = nn.ReLU()
    #     self.bottleneck = bottleneck
    #     if not bottleneck:
    #         self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    # def forward(self, input):
    #     res = self.relu(self.bn1(self.conv1(input)))
    #     res = self.relu(self.bn2(self.conv2(res)))
    #     out = None
    #     if not self.bottleneck:
    #         out = self.pooling(res)
    #     else:
    #         out = res
    #     return out, res

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res



class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()


        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
            
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out
        
class UNet3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super(UNet3D, self).__init__()
        ## original code of 3D U-Net
        # level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        # self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        # self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        # self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        # self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
        # self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        # self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        # self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)


        ### CHANNEL RESIZE
        level_1_chnls, level_2_chnls, level_3_chnls = 4, 16, 32
        bottleneck_channel = 64
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        # self.a_block4 = Conv3DBlock(in_channels=32, out_channels=64)
        # self.a_block5 = Conv3DBlock(in_channels=64, out_channels=128)

        self.bottleNeck = Conv3DBlock(in_channels=32, out_channels=bottleneck_channel, bottleneck= True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)
        ##### CHANNEL RESIZE

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)


        ### CHANNEL RESIZE
        self.linear = nn.Sequential(
            nn.Linear(bottleneck_channel, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
            
        )


    def forward(self, input):
        #Analysis path forward feed
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        # out, residual_level4 = self.a_block4(out)
        # out, residual_level5 = self.a_block5(out)
        out, _ = self.bottleNeck(out)

        # print(out.shape) #(1, 512, 8, 64, 64)
        # quit()
        
        ### For classification output
        out_cls = self.global_avg_pool(out)
        out_cls = torch.flatten(out_cls, 1)
        out_cls = self.linear(out_cls)

        #Synthesis path forward feed
        out_seg = self.s_block3(out, residual_level3)
        out_seg = self.s_block2(out_seg, residual_level2)
        out_seg = self.s_block1(out_seg, residual_level1)

        return out_cls, out_seg



if __name__ == '__main__':
    #Configurations according to the Xenopus kidney dataset
    model = UNet3D(in_channels=1, num_classes=1)
    start_time = time.time()
    # Original configuration of 3D U-Net
    # summary(model=model, input_size=(3, 16, 128, 128), batch_size=-1, device="cpu")
    summary(model=model, input_size=(1, 1, 64, 512, 512), batch_size=-1, device="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))

def get_bounding_boxes(binary_mask):
    """
    Expects binary_mask to be of shape (batch_size, depth, height, width).
    Returns bounding boxes as a tensor of shape (batch_size, depth, 4).
    Each bounding box is represented by (x_min, y_min, x_max, y_max).
    """
    batch_size, _, depth, height, width = binary_mask.shape
    bounding_boxes = []

    for b in range(batch_size):
        boxes_per_batch = []
        for d in range(depth):
            slice_mask = binary_mask[b, 0, d]
            pos = torch.where(slice_mask)
            
            if len(pos[0]) == 0:  # If no foreground pixel is found
                boxes_per_batch.append(torch.tensor([0, 0, width, height], dtype=torch.float32))
            else:
                x_min, y_min = torch.min(pos[1]), torch.min(pos[0])
                x_max, y_max = torch.max(pos[1]), torch.max(pos[0])
                boxes_per_batch.append(torch.tensor([x_min, y_min, x_max-x_min, y_max-y_min], dtype=torch.float32))
        
        bounding_boxes.append(torch.stack(boxes_per_batch))
    
    return torch.stack(bounding_boxes)  # Shape: (batch_size, depth, 4)


class CustomLoss(nn.Module):
    def __init__(self, bbox_lambda=1.0):
        super(CustomLoss, self).__init__()
        self.bbox_lambda = bbox_lambda
        self.bce_loss = nn.BCEWithLogitsLoss()  # Assuming model outputs are logits
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, model_output, target_masks, target_bboxes, device):
        """
        model_output: Raw model outputs (logits) of shape (batch_size, depth, height, width).
        target_masks: Ground truth binary masks of shape (batch_size, depth, height, width).
        target_bboxes: Ground truth bounding boxes of shape (batch_size, depth, 4).
        """
        # Convert model output to binary masks
        pred_masks = torch.sigmoid(model_output)  # Convert logits to probabilities
        pred_masks_bin = (pred_masks > 0.5).float()
        
        # Calculate segmentation loss
        seg_loss = self.bce_loss(model_output, target_masks)
        
        # Extract predicted bounding boxes from binary masks
        pred_bboxes = get_bounding_boxes(pred_masks_bin)
        
        # Calculate bounding box regression loss
        pred_bboxes = pred_bboxes.to(device)
        bbox_loss = self.smooth_l1_loss(pred_bboxes, target_bboxes)
        
        # Combine losses
        total_loss = seg_loss + self.bbox_lambda * bbox_loss
        # return total_loss
        return seg_loss, bbox_loss