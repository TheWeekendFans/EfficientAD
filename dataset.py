import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np

class MVTecDataset(Dataset):
    def __init__(self, root_path, class_name, is_train=True, resize=256, cropsize=256, robustness_aug=False):
        self.root_path = root_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.robustness_aug = robustness_aug
        self.dataset_root = self.resolve_dataset_root()
        
        # Define transforms
        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(cropsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load data paths
        self.x, self.y, self.mask = self.load_dataset_folder()

    def resolve_dataset_root(self):
        candidates = []

        if os.path.basename(os.path.normpath(self.root_path)).lower() == self.class_name.lower():
            candidates.append(self.root_path)

        candidates.append(os.path.join(self.root_path, self.class_name))
        candidates.append(os.path.join(self.root_path, self.class_name, self.class_name))

        for candidate in candidates:
            train_dir = os.path.join(candidate, 'train')
            test_dir = os.path.join(candidate, 'test')
            gt_dir = os.path.join(candidate, 'ground_truth')
            if os.path.isdir(train_dir) and os.path.isdir(test_dir) and os.path.isdir(gt_dir):
                return candidate

        raise FileNotFoundError(
            f"Cannot resolve dataset root for class '{self.class_name}' from root_path='{self.root_path}'. "
            f"Expected one of: <root>/<class>, <root>/<class>/<class>, or direct class folder containing train/test/ground_truth."
        )

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        
        img = Image.open(x).convert('RGB')
        
        # Apply robustness augmentation if enabled (only for good samples in analysis phase)
        if self.robustness_aug and y == 0: 
            img = self.apply_robustness(img)

        img = self.transform_img(img)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = transforms.Resize(self.resize, interpolation=InterpolationMode.NEAREST)(mask)
            mask = transforms.CenterCrop(self.cropsize)(mask)
            mask = transforms.ToTensor()(mask)

        return img, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_root, phase)
        gt_dir = os.path.join(self.dataset_root, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            img_fpath_list = sorted(glob.glob(os.path.join(img_type_dir, '*.png')))
            x.extend(img_fpath_list)

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png') for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        return list(x), list(y), list(mask)

    def apply_robustness(self, img):
        """Simulate production environment variations: lighting and noise."""
        # 1. Random Brightness (Lighting changes)
        # Factor 1.0 is original, <1 is darker, >1 is brighter
        brightness_factor = np.random.uniform(0.8, 1.2)
        img = transforms.functional.adjust_brightness(img, brightness_factor)
        
        # 2. Gaussian Noise (Sensor noise)
        img_np = np.array(img).astype(np.float32)
        noise = np.random.normal(0.0, 15.0, img_np.shape).astype(np.float32)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        return img
