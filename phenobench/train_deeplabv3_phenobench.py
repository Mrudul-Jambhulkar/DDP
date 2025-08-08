import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
import numpy as np
import argparse
import logging
from tqdm import tqdm

# Set up argument parser
parser = argparse.ArgumentParser(description='Train and Test DeepLabV3 on PhenoBench (CPU)')
parser.add_argument('--root', type=str, required=True, help='Path to PhenoBench dataset root')
parser.add_argument('--checkpoint-dir', type=str, default='ckpt/deeplabv3_phenobench', 
                    help='Directory to save checkpoints')
parser.add_argument('--checkpoint', type=str, default=None, 
                    help='Path to checkpoint for testing (skips training if provided)')
parser.add_argument('--skip-training', action='store_true', 
                    help='Skip training phase and only test if checkpoint is provided')
parser.add_argument('--batch-size', type=int, default=16, 
                    help='Batch size (reduce if memory issues occur)')
args = parser.parse_args()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create checkpoint directory if it doesn't exist
os.makedirs(args.checkpoint_dir, exist_ok=True)

# PhenoBench dataset class
class PhenoBench:
    def __init__(self, root: str, split: str = 'train'):
        self.root = root
        self.split = split
        self.image_dir = os.path.join(self.root, split, "images")
        self.semantic_dir = os.path.join(self.root, split, "semantics")
        
        # Verify directories exist
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.semantic_dir):
            raise FileNotFoundError(f"Semantic directory not found: {self.semantic_dir}")
            
        self.filenames = sorted([f for f in os.listdir(self.image_dir) 
                               if f.endswith('.png') or f.endswith('.jpg')])

    def __getitem__(self, index: int) -> dict:
        try:
            sample = {}
            img_path = os.path.join(self.image_dir, self.filenames[index])
            sem_path = os.path.join(self.semantic_dir, self.filenames[index])
            
            sample["image"] = Image.open(img_path).convert("RGB")
            semantic = np.array(Image.open(sem_path))
            
            # Map PhenoBench labels: 3 (partial crops) -> 1 (crops), 4 (partial weeds) -> 2 (weeds)
            semantic[semantic == 3] = 1
            semantic[semantic == 4] = 2
            sample["semantics"] = semantic
            
            return sample
        except Exception as e:
            logger.error(f"Error loading sample {self.filenames[index]}: {str(e)}")
            raise

    def __len__(self):
        return len(self.filenames)

# Custom Dataset Wrapper for DeepLabV3 with resizing to 512x512
class PhenoBenchDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.phenobench = PhenoBench(root=root, split=split)
        
        # Define transform for images
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize image to 512x512
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.phenobench)

    def __getitem__(self, idx):
        sample = self.phenobench[idx]
        image = self.transform(sample['image'])
        
        # Process semantic labels by resizing directly as a tensor
        semantic = torch.tensor(sample['semantics'], dtype=torch.long)
        # Add batch and channel dimensions for F.interpolate: [H, W] -> [1, 1, H, W]
        semantic = semantic.unsqueeze(0).unsqueeze(0)
        # Resize using nearest neighbor interpolation
        semantic = F.interpolate(semantic.float(), size=(512, 512), mode='nearest').long()
        # Remove batch and channel dimensions: [1, 1, H, W] -> [H, W]
        semantic = semantic.squeeze(0).squeeze(0)
        
        return {'rgb': image, 'labels': semantic}

# Compute mIoU and IoU per class
def compute_miou(preds, labels, num_classes, ignore_label=255):
    iou_per_class = []
    class_names = ['background', 'crop', 'weed']  # For PhenoBench classes
    eps = 1e-6  # Small epsilon to avoid division by zero
    
    # Create mask to ignore specific label if needed
    valid_mask = (labels != ignore_label)
    
    for c in range(num_classes):
        pred_c = (preds == c).float() & valid_mask
        target_c = (labels == c).float() & valid_mask
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection + eps

        iou = intersection / union
        iou_per_class.append(iou.item())

    # Calculate mean IoU, ignoring NaN values
    miou = np.nanmean(iou_per_class)
    return miou, iou_per_class

# Main Function
def main():
    # Device configuration (force CPU as requested)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    # Hyperparameters
    num_classes = 3  # background (0), crop (1), weed (2)
    epochs = 5
    learning_rate = 0.001

    # Load DeepLabV3 with ResNet-101 backbone
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)

    # Training Phase (if not skipped)
    if not args.skip_training or args.checkpoint is None:
        logger.info('Starting training phase...')
        train_dataset = PhenoBenchDataset(root=args.root, split='train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for CPU to avoid worker issues
            pin_memory=False
        )

        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # Track best loss
        best_loss = float('inf')
        best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                images = batch['rgb'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                main_loss = criterion(outputs['out'], labels)
                aux_loss = criterion(outputs['aux'], labels)
                loss = main_loss + 0.4 * aux_loss  # Weighted sum of main and aux losses
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            scheduler.step()
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f'Epoch [{epoch+1}/{epochs}] Average Loss: {avg_epoch_loss:.4f}')

            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

            # Save best model if this epoch's loss is the best so far
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }, best_checkpoint_path)
                logger.info(f'New best loss {best_loss:.4f}. Saved best model to: {best_checkpoint_path}')

        logger.info('Training completed.')
    else:
        # Load checkpoint for testing
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided if --skip-training is used")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f'Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get("epoch", "unknown")})')

    # Testing Phase
    logger.info('Starting testing phase...')
    test_dataset = PhenoBenchDataset(root=args.root, split='val')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for CPU to avoid worker issues
        pin_memory=False
    )

    model.eval()
    total_miou = 0.0
    class_ious = [0.0] * num_classes
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing')
        for batch in progress_bar:
            images = batch['rgb'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            batch_miou, batch_ious = compute_miou(preds, labels, num_classes=num_classes)
            total_miou += batch_miou * images.size(0)
            for i in range(num_classes):
                class_ious[i] += batch_ious[i] * images.size(0)
            total_samples += images.size(0)

    # Calculate final metrics
    final_miou = total_miou / total_samples
    final_class_ious = [iou / total_samples for iou in class_ious]
    
    # Log results
    logger.info('-' * 50)
    logger.info('Evaluation Results:')
    logger.info(f'Mean IoU: {final_miou:.4f}')
    for i, (iou, name) in enumerate(zip(final_class_ious, ['background', 'crop', 'weed'])):
        logger.info(f'Class {i} ({name}) IoU: {iou:.4f}')
    logger.info('-' * 50)

if __name__ == '__main__':
    main()