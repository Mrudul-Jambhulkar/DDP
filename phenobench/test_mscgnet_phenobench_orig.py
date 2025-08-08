import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import logging
from collections import OrderedDict
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d
from torch.cuda.amp import GradScaler, autocast
from lib.net.scg_gcn import *

# Setting up argument parser
parser = argparse.ArgumentParser(description='Train and Test MSCG-Net on PhenoBench')
parser.add_argument('--root', type=str, required=True, help='Path to PhenoBench dataset root')
parser.add_argument('--checkpoint-dir', type=str, default='ckpt/phenobench_mscgnet101_orig', help='Directory to save/load checkpoints')
parser.add_argument('--log-dir', type=str, default='ckpt/phenobench_mscgnet101_orig/logs', help='Directory to save logs')
parser.add_argument('--skip-training', action='store_true', help='Skip training and only test the model')
parser.add_argument('--checkpoint', type=str, default=os.path.join('ckpt', 'phenobench_mscgnet101_orig', 'best_model.pth'), help='Path to model checkpoint for testing')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size per GPU (default: 2)')
args = parser.parse_args()

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create checkpoint and log directories
os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

# Clear GPU memory
torch.cuda.empty_cache()

# ACW_loss class (unchanged)
class ACW_loss(nn.Module):
    def __init__(self, ini_weight=0, ini_iteration=0, eps=1e-4, ignore_index=255):
        super(ACW_loss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = ini_weight
        self.itr = ini_iteration
        self.eps = eps

    def forward(self, prediction, target):
        pred = F.softmax(prediction, 1)
        one_hot_label, mask = self.encode_one_hot_label(pred, target)
        acw = self.adaptive_class_weight(pred, one_hot_label, mask)
        err = torch.pow((one_hot_label - pred), 2)
        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc, 1)
        intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
        union = pred + one_hot_label
        if mask is not None:
            union[mask] = 0
        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union
        dice = torch.clamp(dice, min=self.eps)
        return loss_pnc.mean() - dice.mean().log()

    def adaptive_class_weight(self, pred, one_hot_label, mask=None):
        self.itr += 1
        sum_class = torch.sum(one_hot_label, dim=(0, 2, 3))
        sum_norm = sum_class / sum_class.sum()
        self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / mfb.sum()
        mfb = torch.clamp(mfb, min=0.001, max=1.0)
        acw = (1. + pred + one_hot_label) * mfb.unsqueeze(-1).unsqueeze(-1)
        if mask is not None:
            acw[mask] = 0
        return acw

    def encode_one_hot_label(self, pred, target):
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            return one_hot_label, None

# PhenoBench class (unchanged)
class PhenoBench:
    def __init__(self, root: str, split: str = 'train', target_types: list = ['semantics'], 
                 ignore_partial: bool = False, make_unique_ids: bool = True, ignore_mask: int = 255) -> None:
        root = os.path.expanduser(root)
        assert os.path.exists(root), f"The path to the dataset does not exist: `{root}`."
        assert split in ['train', 'val', 'test']
        for target_type in target_types:
            assert target_type in ['semantics', 'plant_instances', 'leaf_instances', 'plant_bboxes', 
                                   'leaf_bboxes', 'plant_visibility', 'leaf_visibility']
        self.root = root
        self.split = split
        self.target_types = target_types
        self.ignore_partial = ignore_partial
        self.make_unique_ids = make_unique_ids
        self.ignore_mask = ignore_mask
        self.filenames = sorted(os.listdir(os.path.join(self.root, split, "images")))

    def __getitem__(self, index: int) -> dict:
        sample = {}
        sample["image_name"] = self.filenames[index]
        sample["image"] = Image.open(os.path.join(self.root, self.split, "images", self.filenames[index])).convert("RGB")
        if self.split in ["train", "val"]:
            for target in self.target_types:
                if target in ["semantics", "plant_instances", "leaf_instances", "plant_visibility", "leaf_visibility"]:
                    sample[target] = np.array(Image.open(os.path.join(self.root, self.split, target, self.filenames[index])))
            if self.ignore_partial:
                semantics = np.array(Image.open(os.path.join(self.root, self.split, "semantics", self.filenames[index])))
                partial_crops = semantics == 3
                partial_weeds = semantics == 4
                if "semantics" in self.target_types:
                    sample["semantics"][partial_crops] = self.ignore_mask
                    sample["semantics"][partial_weeds] = self.ignore_mask
                for target in self.target_types:
                    if target in ["plant_instances", "leaf_instances"]:
                        sample[target][partial_crops] = 0
                        sample[target][partial_weeds] = 0
                    if target in ["plant_bboxes", "leaf_bboxes"]:
                        pass
            else:
                if "semantics" in self.target_types:
                    sample["semantics"][sample["semantics"] == 3] = 1
                    sample["semantics"][sample["semantics"] == 4] = 2
                if "plant_bboxes" in self.target_types:
                    semantics = np.array(Image.open(os.path.join(self.root, self.split, "semantics", self.filenames[index])))
                    plant_instances = np.array(Image.open(os.path.join(self.root, self.split, "plant_instances", self.filenames[index])))
                    leaf_instances = np.array(Image.open(os.path.join(self.root, self.split, "leaf_instances", self.filenames[index])))
                    sample["plant_bboxes"] = []
                    for label in [1, 2]:
                        for plant_id in np.unique(plant_instances[semantics == label]):
                            ys, xs = np.where((plant_instances == plant_id) & (semantics == label))
                            width, height = np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)
                            center = (np.min(xs) + width // 2, np.min(ys) + height // 2)
                            sample["plant_bboxes"].append({
                                "label": label,
                                "corner": (np.min(xs), np.min(ys)),
                                "center": center,
                                "width": width,
                                "height": height
                            })
                if "leaf_bboxes" in self.target_types:
                    semantics = np.array(Image.open(os.path.join(self.root, self.split, "semantics", self.filenames[index])))
                    leaf_instances = np.array(Image.open(os.path.join(self.root, self.split, "leaf_instances", self.filenames[index])))
                    sample["leaf_bboxes"] = []
                    for leaf_id in np.unique(leaf_instances[leaf_instances > 0]):
                        ys, xs = np.where((leaf_instances == leaf_id))
                        width, height = np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)
                        center = (np.min(xs) + width // 2, np.min(ys) + height // 2)
                        sample["leaf_bboxes"].append({
                            "label": 1,
                            "corner": (np.min(xs), np.min(ys)),
                            "center": center,
                            "width": width,
                            "height": height
                        })
        if self.make_unique_ids:
            def replace(array: np.array, values, replacements):
                temp_array = array.copy()
                for v, r in zip(values, replacements):
                    temp_array[array == v] = r
                array = temp_array
            if "plant_instances" in self.target_types:
                semantics = np.array(Image.open(os.path.join(self.root, self.split, "semantics", self.filenames[index])))
                crop_ids = np.unique(sample["plant_instances"][semantics == 1])
                weed_ids = np.unique(sample["plant_instances"][semantics == 2])
                N, M = len(crop_ids), len(weed_ids)
                replace(sample["plant_instances"][semantics == 1], crop_ids, np.arange(1, N + 1))
                replace(sample["plant_instances"][semantics == 2], weed_ids, np.arange(N + 1, N + M + 1))
            if "leaf_instances" in self.target_types:
                leaf_ids = np.unique(sample["leaf_instances"][sample["leaf_instances"] > 0])
                replace(sample["leaf_instances"], leaf_ids, np.arange(1, len(leaf_ids) + 1))
        return sample

    def __len__(self):
        return len(self.filenames)

# PhenoBenchDataset (updated for DataParallel)
class PhenoBenchDataset(Dataset):
    def __init__(self, root, split, modalities, transform=None):
        self.root = root
        self.split = split
        self.modalities = modalities
        self.transform = transform
        self.phenobench = PhenoBench(
            root=root,
            split=split,
            target_types=['semantics'],
            ignore_partial=False,
            make_unique_ids=False,
            ignore_mask=255
        )

    def __len__(self):
        return len(self.phenobench)

    def __getitem__(self, idx):
        sample = self.phenobench[idx]
        image = sample['image']
        semantic = sample['semantics']
        if self.transform:
            image = transforms.ToTensor()(image)
            semantic = torch.tensor(semantic, dtype=torch.long)
            if self.split == 'train':
                scale = np.random.uniform(self.transform['scale_min'], self.transform['scale_max'])
                new_size = int(1024 * scale)
                image = transforms.Resize((new_size, new_size))(image)
                semantic = semantic.unsqueeze(0)
                semantic = transforms.Resize((new_size, new_size), interpolation=transforms.InterpolationMode.NEAREST)(semantic)
                semantic = semantic.squeeze(0)
                angle = np.random.uniform(self.transform['rotate_min'], self.transform['rotate_max'])
                image = transforms.functional.rotate(image, angle)
                semantic = semantic.unsqueeze(0)
                semantic = transforms.functional.rotate(semantic, angle)
                semantic = semantic.squeeze(0)
                image = transforms.Resize((1024, 1024))(image)
                semantic = semantic.unsqueeze(0)
                semantic = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.NEAREST)(semantic)
                semantic = semantic.squeeze(0)
        else:
            image = transforms.ToTensor()(image)
            semantic = torch.tensor(semantic, dtype=torch.long)
        return {'rgb': image, 'labels': semantic}

# mIoU computation (unchanged)
def compute_miou(preds, labels, num_classes, ignore_label=255):
    valid_mask = (labels != ignore_label)
    iou_per_class = []
    for c in range(num_classes):
        pred_c = (preds == c) & valid_mask
        target_c = (labels == c) & valid_mask
        intersection = (pred_c & target_c).float().sum()
        union = (pred_c | target_c).float().sum()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append(intersection / union)
    iou_per_class = np.array(iou_per_class)
    valid_iou = iou_per_class[~np.isnan(iou_per_class)]
    mean_iou = np.mean(valid_iou) if len(valid_iou) > 0 else 0.0
    return mean_iou, iou_per_class

# Main function (updated with mixed precision and DataParallel)
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info(f'Available GPUs: {torch.cuda.device_count()}')

    num_classes = 3
    class_names = ["background", "crop", "weed"]
    batch_size = args.batch_size
    epochs = 25
    base_lr = 0.01
    momentum = 0.9
    weight_decay = 0.0001

    augmentation = {
        'scale_min': 0.5,
        'scale_max': 2.0,
        'rotate_min': -10,
        'rotate_max': 10,
        'ignore_label': 255
    }

    scaler = GradScaler()

    if not args.skip_training:
        logger.info('Starting training phase...')
        train_dataset = PhenoBenchDataset(
            root=args.root,
            split='train',
            modalities=['rgb'],
            transform=augmentation
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        model = load_model(name='MSCG-Rx101', classes=num_classes, node_size=(32, 32))
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        criterion = ACW_loss(ignore_index=255)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=epochs * len(train_loader),
            power=0.9
        )
        best_loss = float('inf')
        best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, batch in enumerate(train_loader):
                images = batch['rgb'].to(device)
                labels = batch['labels'].to(device)
                if i == 0:
                    logger.info(f"Batch {i+1} - Image shape: {images.shape}, Labels shape: {labels.shape}")
                    logger.info(f"Batch {i+1} - Label values: {labels.unique()}")
                optimizer.zero_grad()
                with autocast():
                    outputs, aux_loss = model(images)
                    main_loss = criterion(outputs, labels)
                    total_loss = main_loss + aux_loss
                if torch.isnan(total_loss):
                    logger.warning(f"NaN loss detected at Epoch {epoch+1}, Step {i+1}. Skipping this batch.")
                    continue
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_loss += total_loss.item()
                if i % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}')
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f'Epoch [{epoch+1}/{epochs}] Average Loss: {avg_epoch_loss:.4f}')
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save({'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()}, checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')
            if avg_epoch_loss < best_loss and not np.isnan(avg_epoch_loss):
                best_loss = avg_epoch_loss
                torch.save({'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()}, best_checkpoint_path)
                logger.info(f'New best loss {best_loss:.4f}. Saved best model to: {best_checkpoint_path}')
        logger.info('Training completed.')

    logger.info('Starting testing phase...')
    test_dataset = PhenoBenchDataset(
        root=args.root,
        split='val',
        modalities=['rgb'],
        transform=None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    model = load_model(name='MSCG-Rx101', classes=num_classes, node_size=(32, 32))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    logger.info(f'Loaded checkpoint: {args.checkpoint}')
    logger.info("Classes considered:")
    for idx, class_name in enumerate(class_names):
        logger.info(f"Class {idx}: {class_name}")
    total_miou = 0.0
    total_iou_per_class = np.zeros(num_classes)
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['rgb'].to(device)
            labels = batch['labels'].to(device)
            with autocast():
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            batch_miou, batch_iou_per_class = compute_miou(preds, labels, num_classes=num_classes, ignore_label=255)
            total_miou += batch_miou * images.size(0)
            total_iou_per_class += batch_iou_per_class * images.size(0)
            total_samples += images.size(0)
    total_miou = total_miou / total_samples if total_samples > 0 else 0.0
    total_iou_per_class = total_iou_per_class / total_samples if total_samples > 0 else np.zeros(num_classes)
    logger.info("Per-class IoU on validation set:")
    for class_idx, class_name in enumerate(class_names):
        iou = total_iou_per_class[class_idx]
        if np.isnan(iou):
            logger.info(f"{class_name}: NaN (no instances in validation set)")
        else:
            logger.info(f"{class_name}: {iou:.4f}")
    logger.info(f"Mean IoU on validation set: {total_miou:.4f}")

if __name__ == '__main__':
    main()