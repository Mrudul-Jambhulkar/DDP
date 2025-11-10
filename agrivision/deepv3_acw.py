import os
import sys
import glob
import math
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.optim as optim



# Add the DeepLab path

xception_path = "/home/agipml/21d070044/Agriculture-Vision-2021/pytorch-deeplab-xception-master/pytorch-deeplab-xception-master"

# Add it to sys.path
sys.path.append(xception_path)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

BatchNorm2d = SynchronizedBatchNorm2d

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self)._init_()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
            rep.append(BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2))

        if stride == 1 and is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))


        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip

        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16, pretrained=False):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True,
                            is_last=True)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn3 = BatchNorm2d(1536)

        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn4 = BatchNorm2d(1536)

        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn5 = BatchNorm2d(2048)

        # Init weights
        self._init_weight()

        # Load pretrained model
        if pretrained:
            self._load_xception_pretrained()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_xception_pretrained(self):
        local_weight_path = "/home/agipml/21d070044/Agriculture-Vision-2021/xception-b5690688.pth"
        pretrain_dict = torch.load(local_weight_path)
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in model_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, freeze_bn=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Backbone: Xception")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.xception_features = Xception(nInputChannels, os, pretrained)

        # ASPP
        if os == 16:
            dilations = [1, 6, 12, 18]
        elif os == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, dilation=dilations[0])
        self.aspp2 = ASPP_module(2048, 256, dilation=dilations[1])
        self.aspp3 = ASPP_module(2048, 256, dilation=dilations[2])
        self.aspp4 = ASPP_module(2048, 256, dilation=dilations[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        if freeze_bn:
            self._freeze_bn()

    def forward(self, input):
        x, low_level_features = self.xception_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, BatchNorm2d):
                m.eval()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.xception_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


def adapt_first_conv_layer(model):
    w = model.xception_features.conv1.weight.data.clone()  # shape: (out_channels, 3, kH, kW)

    w_new = torch.zeros(w.size(0), 4, w.size(2), w.size(3))  # new weights for 4 channels

    # Duplicate Red channel weights for NIR initialization
    w_new[:, 0, :, :] = w[:, 0, :, :]  # NIR channel
    w_new[:, 1, :, :] = w[:, 0, :, :]  # Red
    w_new[:, 2, :, :] = w[:, 1, :, :]  # Green
    w_new[:, 3, :, :] = w[:, 2, :, :]  # Blue

    # Replace conv1 weights and input channels
    new_conv = torch.nn.Conv2d(4, w.size(0), kernel_size=w.size(2),
                               stride=model.xception_features.conv1.stride,
                               padding=model.xception_features.conv1.padding,
                               bias=False)

    new_conv.weight.data = w_new
    model.xception_features.conv1 = new_conv



# ==================== ACW LOSS (Simple Version) ====================
class ACW_MultiLabel_loss(nn.Module):
    def __init__(self, n_classes=9, eps=1e-5):
        super(ACW_MultiLabel_loss, self).__init__()
        self.n_classes = n_classes
        self.eps = eps
        self.weight = torch.ones(n_classes)
        self.itr = 0

    def forward(self, prediction, target, valid_mask=None):
        # Apply sigmoid for multi-label
        pred = torch.sigmoid(prediction)
        
        # Apply valid mask
        if valid_mask is not None:
            pred = pred * valid_mask
            target = target * valid_mask
        
        # Calculate adaptive weights
        acw = self.adaptive_class_weight(pred, target, valid_mask)
        
        # PNC term
        err = torch.pow((target - pred), 2)
        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc) / (torch.sum(acw) + self.eps)
        
        # Dice term
        intersection = 2 * torch.sum(pred * target, dim=(0, 2, 3)) + self.eps
        union = torch.sum(pred, dim=(0, 2, 3)) + torch.sum(target, dim=(0, 2, 3)) + self.eps
        dice = intersection / union
        dice_loss = 1 - dice.mean()
        
        return loss_pnc + dice_loss

    def adaptive_class_weight(self, pred, target, valid_mask=None):
        self.itr += 1
        
        if valid_mask is not None:
            sum_class = torch.sum(target * valid_mask, dim=(0, 2, 3))
            total_valid = torch.sum(valid_mask, dim=(0, 2, 3)) + self.eps
            sum_norm = sum_class / total_valid
        else:
            sum_class = torch.sum(target, dim=(0, 2, 3))
            total_pixels = target.shape[0] * target.shape[2] * target.shape[3]
            sum_norm = sum_class / total_pixels
        
        if self.itr == 1:
            self.weight = sum_norm.detach()
        else:
            self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr
        
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / mfb.sum()
        mfb = torch.clamp(mfb, min=0.001, max=1.0)
        
        acw = (1. + pred + target) * mfb.view(1, self.n_classes, 1, 1)
        
        if valid_mask is not None:
            acw = acw * valid_mask
            
        return acw

# ==================== DATASET CLASS ====================
class AgriVisionDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.rgb_images = sorted(glob.glob(os.path.join(root_dir, 'images', 'rgb', '*.jpg')))
        self.nir_images = sorted(glob.glob(os.path.join(root_dir, 'images', 'nir', '*.jpg')))

        self.classes = ['double_plant', 'drydown', 'endrow', 'nutrient_deficiency',
                        'planter_skip', 'storm_damage', 'water', 'waterway', 'weed_cluster']
        self.labels_paths = {cls: sorted(glob.glob(os.path.join(root_dir, 'labels', cls, '*.png'))) for cls in self.classes}

        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, 'masks', '*.png')))
        self.boundary_paths = sorted(glob.glob(os.path.join(root_dir, 'boundaries', '*.png')))

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_img = Image.open(self.rgb_images[idx]).convert('RGB')
        nir_img = Image.open(self.nir_images[idx]).convert('L')

        rgb_tensor = transforms.ToTensor()(rgb_img)  # [3,512,512]
        nir_tensor = transforms.ToTensor()(nir_img)  # [1,512,512]
        image = torch.cat([rgb_tensor, nir_tensor], dim=0)  # 4 channels

        label_masks = []
        for cls in self.classes:
            lbl_img = Image.open(self.labels_paths[cls][idx]).convert('L')
            lbl_tensor = transforms.ToTensor()(lbl_img)
            label_masks.append(lbl_tensor)
        labels = torch.cat(label_masks, dim=0)

        mask_img = Image.open(self.mask_paths[idx]).convert('L')
        boundary_img = Image.open(self.boundary_paths[idx]).convert('L')
        mask = transforms.ToTensor()(mask_img)
        boundary = transforms.ToTensor()(boundary_img)
        valid_mask = (mask > 0.5).float() * (boundary > 0.5).float()

        sample = {
            'image': image.float(),
            'labels': labels.float(),
            'valid_mask': valid_mask.float()
        }
        return sample

# ==================== VALIDATION FUNCTION ====================
def validate_model(model, val_loader, device):
    """Calculate validation loss and per-class IoU"""
    model.eval()
    val_loss = 0.0
    acw_criterion = ACW_MultiLabel_loss(n_classes=9)
    
    # For IoU calculation
    total_intersection = torch.zeros(9).to(device)
    total_union = torch.zeros(9).to(device)
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['image'].to(device)
            labels = batch['labels'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            outputs = model(inputs)
            loss = acw_criterion(outputs, labels, valid_mask)
            val_loss += loss.item()
            
            # Calculate IoU for this batch
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            # Apply valid mask
            predictions = predictions * valid_mask
            labels = labels * valid_mask
            
            intersection = (predictions * labels).sum(dim=(0, 2, 3))
            union = predictions.sum(dim=(0, 2, 3)) + labels.sum(dim=(0, 2, 3)) - intersection
            
            total_intersection += intersection
            total_union += union
    
    # Calculate per-class IoU
    iou_per_class = total_intersection / (total_union + 1e-6)
    mean_iou = iou_per_class.mean()
    
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, mean_iou, iou_per_class

# ==================== TRAINING WITH VALIDATION ====================
import os
import time
from datetime import datetime
import torch
import torch.optim as optim
from tqdm import tqdm
import glob

def train(model, train_loader, val_loader, optimizer, device, num_epochs=30, 
          save_path='deepv3bestacw.pt', resume_checkpoint=None):
    """
    Train model with checkpoint saving and resuming capability
    
    Args:
        resume_checkpoint: Path to checkpoint file to resume from
    """
    
    # Create checkpoint directory with timestamp
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoint_{start_time}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory created: {checkpoint_dir}")
    
    best_val_loss = float('inf')
    start_epoch = 0
    model.to(device)
    print(f'Training on device: {device}')

    acw_criterion = ACW_MultiLabel_loss(n_classes=9)
    
    # Class names for display
    class_names = ['double_plant', 'drydown', 'endrow', 'nutrient_deficiency',
                   'planter_skip', 'storm_damage', 'water', 'waterway', 'weed_cluster']

    # Resume from checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed training from epoch {start_epoch}")
        print(f"Previous best val loss: {best_val_loss:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

        for batch in loop:
            inputs = batch['image'].to(device)
            labels = batch['labels'].to(device)
            valid_mask = batch['valid_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = acw_criterion(outputs, labels, valid_mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(train_loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        avg_val_loss, mean_iou, iou_per_class = validate_model(model, val_loader, device)
        
        # Print results with per-class IoU
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val mIoU: {mean_iou:.4f}')
        print('Per-class IoU:')
        for i, name in enumerate(class_names):
            print(f'  {name:<20}: {iou_per_class[i]:.4f}')
        print('-' * 60)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1}_valloss{avg_val_loss:.4f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'mean_iou': mean_iou,
            }, checkpoint_path)
            print(f'Saved checkpoint at epoch {epoch+1}: {checkpoint_path}')

        # Save best model based on validation loss (DELETE previous and save new with epoch number)
        if avg_val_loss < best_val_loss:
            # Delete previous best model files
            previous_best_files = glob.glob(os.path.join(checkpoint_dir, 'bestmodel_epoch*_valloss*.pt'))
            for file_path in previous_best_files:
                try:
                    os.remove(file_path)
                    print(f"Deleted previous best model: {os.path.basename(file_path)}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
            
            # Update best validation loss
            best_val_loss = avg_val_loss
            
            # Save new best model with epoch number
            best_model_path = os.path.join(checkpoint_dir, f'bestmodel_epoch{epoch+1}_valloss{avg_val_loss:.4f}.pt')
            torch.save(model.state_dict(), best_model_path)
            
            # Also save the main model file (original behavior)
            torch.save(model.state_dict(), save_path)
            
            print(f'Saved NEW BEST model at epoch {epoch+1} with val loss: {best_val_loss:.4f}')
            print(f'Best model saved as: {best_model_path}')

    print(f"Training completed. All checkpoints saved in: {checkpoint_dir}")

# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ model')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint file to resume training from')
    args = parser.parse_args()

    # Paths
    train_root_dir = '/home/agipml/21d070044/Agriculture-Vision-2021/train'
    val_root_dir = '/home/agipml/21d070044/Agriculture-Vision-2021/val'

    # Create datasets
    train_dataset = AgriVisionDataset(train_root_dir)
    val_dataset = AgriVisionDataset(val_root_dir)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model setup (assuming these are defined elsewhere)
    model = DeepLabv3_plus(nInputChannels=4, n_classes=9, os=16, pretrained=True, _print=True)
    adapt_first_conv_layer(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    optimizer = torch.optim.Adam([
        {'params': get_1x_lr_params(model), 'lr': 1e-4},
        {'params': get_10x_lr_params(model), 'lr': 1e-3},
    ])

    # Train with validation and optional resume
    train(model, train_loader, val_loader, optimizer, device, 
          num_epochs=100, 
          save_path='deepv3acw_final.pt',
          resume_checkpoint=args.resume)