import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset
from model import Teacher, Student, AutoEncoder, FEATURE_CHANNELS
from runtime_config import DEFAULTS, get_run_presets


USE_EMBEDDED_CONFIG = True
EMBEDDED_CLASS = 'grid'
EMBEDDED_OVERRIDES = {
    # 'epochs': 100,
    # 'lr': 5e-5,
}

PYRAMID_LAYERS = ['layer1', 'layer2', 'layer3']
DEFAULT_HARD_KEEP_RATIO = DEFAULTS.get('hard_keep_ratio', 0.1)
DEFAULT_SEP_WEIGHT = DEFAULTS.get('sep_weight', 0.3)
DEFAULT_SEP_MARGIN = DEFAULTS.get('sep_margin', 0.25)


def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def compute_teacher_feature_stats(teacher, loader, device):
    teacher.eval()
    stats_acc = {}

    for layer in PYRAMID_LAYERS:
        channels = FEATURE_CHANNELS[layer]
        stats_acc[layer] = {
            'sum': torch.zeros(channels, device=device),
            'sumsq': torch.zeros(channels, device=device),
            'count': 0,
        }

    with torch.no_grad():
        for img, _, _ in tqdm(loader, desc='Computing teacher feature stats'):
            img = img.to(device, non_blocking=True)
            feats = teacher(img)

            for layer in PYRAMID_LAYERS:
                feat = feats[layer].float()
                b, c, h, w = feat.shape
                feat_flat = feat.permute(1, 0, 2, 3).reshape(c, -1)
                stats_acc[layer]['sum'] += feat_flat.sum(dim=1)
                stats_acc[layer]['sumsq'] += (feat_flat ** 2).sum(dim=1)
                stats_acc[layer]['count'] += b * h * w

    mean_std = {}
    for layer in PYRAMID_LAYERS:
        layer_sum = stats_acc[layer]['sum']
        layer_sumsq = stats_acc[layer]['sumsq']
        count = max(1, stats_acc[layer]['count'])
        mean = layer_sum / count
        var = layer_sumsq / count - mean ** 2
        var = torch.clamp(var, min=1e-6)
        std = torch.sqrt(var)
        mean_std[layer] = {
            'mean': mean.view(1, -1, 1, 1),
            'std': std.view(1, -1, 1, 1),
        }

    return mean_std


def save_feature_stats(path, mean_std):
    payload = {}
    for layer in PYRAMID_LAYERS:
        payload[f'{layer}_mean'] = mean_std[layer]['mean'].detach().cpu().numpy()
        payload[f'{layer}_std'] = mean_std[layer]['std'].detach().cpu().numpy()
    np.savez(path, **payload)


def normalize_teacher_features(feats, mean_std):
    norm = {}
    for layer in PYRAMID_LAYERS:
        mean = mean_std[layer]['mean']
        std = mean_std[layer]['std']
        norm[layer] = (feats[layer] - mean) / (std + 1e-6)
    return norm


def hard_mining_loss(err_map, keep_ratio=0.1):
    flat = err_map.flatten(start_dim=1)
    n = flat.shape[1]
    k = max(1, int(n * keep_ratio))
    top_vals = torch.topk(flat, k, dim=1, largest=True).values
    return top_vals.mean()


def synthesize_anomaly_batch(img):
    b, _, h, w = img.shape
    aug = img.clone()
    mask = torch.zeros((b, 1, h, w), device=img.device)

    for i in range(b):
        patch_h = np.random.randint(max(8, h // 16), max(16, h // 4))
        patch_w = np.random.randint(max(8, w // 16), max(16, w // 4))
        y1 = np.random.randint(0, max(1, h - patch_h))
        x1 = np.random.randint(0, max(1, w - patch_w))

        src_y = np.random.randint(0, max(1, h - patch_h))
        src_x = np.random.randint(0, max(1, w - patch_w))

        patch = aug[i:i + 1, :, src_y:src_y + patch_h, src_x:src_x + patch_w].clone()
        noise = torch.randn_like(patch) * 0.15
        patch = torch.clamp(patch + noise, min=-3.0, max=3.0)

        aug[i:i + 1, :, y1:y1 + patch_h, x1:x1 + patch_w] = patch
        mask[i:i + 1, :, y1:y1 + patch_h, x1:x1 + patch_w] = 1.0

    jitter = torch.empty((b, 1, 1, 1), device=img.device).uniform_(0.9, 1.1)
    aug = aug * jitter
    return aug, mask


def build_layer2_memory_bank(teacher, loader, device, mean_std, max_patches=20000):
    teacher.eval()
    patches = []

    with torch.no_grad():
        for img, _, _ in tqdm(loader, desc='Building layer2 memory bank'):
            img = img.to(device, non_blocking=True)
            feats = teacher(img)
            feats = normalize_teacher_features(feats, mean_std)
            layer2 = feats['layer2']
            patch = layer2.permute(0, 2, 3, 1).reshape(-1, layer2.shape[1])
            patches.append(patch.detach().cpu())

    if len(patches) == 0:
        raise RuntimeError('No patches collected for memory bank.')

    bank = torch.cat(patches, dim=0)
    if bank.shape[0] > max_patches:
        idx = torch.randperm(bank.shape[0])[:max_patches]
        bank = bank[idx]

    return bank.numpy().astype(np.float32)


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'models', args.class_name)
    os.makedirs(model_dir, exist_ok=True)

    device = resolve_device(getattr(args, 'device', 'auto'))
    print(f'Using device: {device}')

    train_dataset = MVTecDataset(args.data_path, args.class_name, is_train=True)
    num_workers = getattr(args, 'num_workers', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    stats_loader = DataLoader(
        train_dataset,
        batch_size=max(1, min(args.batch_size, 8)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    teacher = Teacher(args.pdn_size).to(device)
    student = Student(args.pdn_size).to(device)
    autoencoder = AutoEncoder(out_channels=FEATURE_CHANNELS['layer3']).to(device)

    if args.teacher_path and os.path.exists(args.teacher_path):
        teacher.load_pretrained_teacher(args.teacher_path, map_location=device)
        print(f'Loaded teacher weights from {args.teacher_path}')

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    teacher_ckpt_path = os.path.join(model_dir, 'teacher_initial.pth')
    torch.save(teacher.state_dict(), teacher_ckpt_path)

    feature_stats_path = os.path.join(model_dir, 'feature_stats.npz')
    mean_std = compute_teacher_feature_stats(teacher, stats_loader, device)
    save_feature_stats(feature_stats_path, mean_std)

    memory_bank = build_layer2_memory_bank(
        teacher,
        stats_loader,
        device,
        mean_std,
        max_patches=int(getattr(args, 'bank_size', 20000)),
    )
    np.save(os.path.join(model_dir, 'memory_bank_layer2.npy'), memory_bank)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(autoencoder.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    use_amp = bool(getattr(args, 'amp', False)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if str(getattr(args, 'class_name', '')).lower() == 'grid':
        layer_weights = {'layer1': 0.2, 'layer2': 0.7, 'layer3': 0.1}
    else:
        layer_weights = {'layer1': 0.5, 'layer2': 0.3, 'layer3': 0.2}
    hard_keep_ratio = float(getattr(args, 'hard_keep_ratio', 0.1))
    sep_weight = float(getattr(args, 'sep_weight', 0.3))
    margin = float(getattr(args, 'sep_margin', 0.25))

    print(f'Starting training for class {args.class_name}...')
    for epoch in range(args.epochs):
        student.train()
        autoencoder.train()

        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for img, _, _ in pbar:
            img = img.to(device, non_blocking=True)
            img_aug, aug_mask = synthesize_anomaly_batch(img)

            with torch.no_grad():
                teacher_feats = teacher(img)
                teacher_feats = normalize_teacher_features(teacher_feats, mean_std)

                teacher_feats_aug = teacher(img_aug)
                teacher_feats_aug = normalize_teacher_features(teacher_feats_aug, mean_std)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                student_feats = student(img)
                student_feats_aug = student(img_aug)

                loss_st = torch.tensor(0.0, device=device)
                for layer in PYRAMID_LAYERS:
                    err_map = (teacher_feats[layer] - student_feats[layer]) ** 2
                    err_map = torch.mean(err_map, dim=1, keepdim=True)
                    loss_layer = hard_mining_loss(err_map, keep_ratio=hard_keep_ratio)
                    loss_st = loss_st + layer_weights[layer] * loss_layer

                ae_out = autoencoder(teacher_feats['layer3'].detach())
                loss_ae = F.mse_loss(ae_out, teacher_feats['layer3'])

                ae_out_aug = autoencoder(teacher_feats_aug['layer3'].detach())

                err_aug = torch.tensor(0.0, device=device)
                for layer in PYRAMID_LAYERS:
                    layer_err = torch.mean((teacher_feats_aug[layer] - student_feats_aug[layer]) ** 2, dim=1, keepdim=True)
                    layer_err = F.interpolate(layer_err, size=(img.shape[-2], img.shape[-1]), mode='bilinear', align_corners=False)
                    err_aug = err_aug + layer_weights[layer] * layer_err

                ae_err_aug = torch.mean((teacher_feats_aug['layer3'] - ae_out_aug) ** 2, dim=1, keepdim=True)
                ae_err_aug = F.interpolate(ae_err_aug, size=(img.shape[-2], img.shape[-1]), mode='bilinear', align_corners=False)
                anomaly_aug = 0.7 * err_aug + 0.3 * ae_err_aug

                pos_score = (anomaly_aug * aug_mask).sum(dim=(1, 2, 3)) / (aug_mask.sum(dim=(1, 2, 3)) + 1e-6)
                neg_mask = 1.0 - aug_mask
                neg_score = (anomaly_aug * neg_mask).sum(dim=(1, 2, 3)) / (neg_mask.sum(dim=(1, 2, 3)) + 1e-6)
                loss_sep = torch.relu(margin + neg_score - pos_score).mean()

                total_loss = loss_st + 0.5 * loss_ae + sep_weight * loss_sep

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{epoch_loss / max(1, n_batches):.6f}'})

        scheduler.step()

        save_every = max(1, int(getattr(args, 'save_every', 10)))
        if (epoch + 1) % save_every == 0 or (epoch + 1) == args.epochs:
            torch.save(student.state_dict(), os.path.join(model_dir, f'student_epoch_{epoch + 1}.pth'))
            torch.save(autoencoder.state_dict(), os.path.join(model_dir, f'autoencoder_epoch_{epoch + 1}.pth'))

        torch.save(student.state_dict(), os.path.join(model_dir, 'student_last.pth'))
        torch.save(autoencoder.state_dict(), os.path.join(model_dir, 'autoencoder_last.pth'))

    print('Training finished.')


if __name__ == '__main__':
    if USE_EMBEDDED_CONFIG:
        presets = get_run_presets()
        args = presets[f'train_{EMBEDDED_CLASS}']
        for key, value in EMBEDDED_OVERRIDES.items():
            setattr(args, key, value)
        print('Using embedded config mode in train.py')
        train(args)
        raise SystemExit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DEFAULTS['data_path'])
    parser.add_argument('--class_name', type=str, default=DEFAULTS['class_name'])
    parser.add_argument('--output_dir', type=str, default=DEFAULTS['output_dir'])
    parser.add_argument('--pdn_size', type=str, default=DEFAULTS['pdn_size'], choices=['small', 'medium'])
    parser.add_argument('--teacher_path', type=str, default=DEFAULTS['teacher_path'])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--num_workers', type=int, default=DEFAULTS['num_workers'])
    parser.add_argument('--device', type=str, default=DEFAULTS['device'])
    parser.add_argument('--amp', action='store_true', default=DEFAULTS['amp'])
    parser.add_argument('--save_every', type=int, default=DEFAULTS['save_every'])
    parser.add_argument('--hard_keep_ratio', type=float, default=DEFAULT_HARD_KEEP_RATIO)
    parser.add_argument('--sep_weight', type=float, default=DEFAULT_SEP_WEIGHT)
    parser.add_argument('--sep_margin', type=float, default=DEFAULT_SEP_MARGIN)
    parser.add_argument('--bank_size', type=int, default=DEFAULTS.get('bank_size', 20000))
    args = parser.parse_args()

    train(args)
