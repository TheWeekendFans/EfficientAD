import argparse
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset
from model import Teacher, Student, AutoEncoder, FEATURE_CHANNELS
from runtime_config import DEFAULTS, get_run_presets


USE_EMBEDDED_CONFIG = True
EMBEDDED_CLASS = 'grid'
EMBEDDED_OVERRIDES = {
    # 'epoch': 100,
}

PYRAMID_LAYERS = ['layer1', 'layer2', 'layer3']
LAYER_WEIGHTS = {'layer1': 0.5, 'layer2': 0.3, 'layer3': 0.2}


def get_fusion_weights(args):
    class_name = str(getattr(args, 'class_name', '')).lower()
    if class_name == 'grid':
        return {
            'layer1': 0.2,
            'layer2': 0.8,
            'layer3': 0.0,
            'ae': 0.1,
            'st': 0.6,
            'bank': 0.4,
        }
    return {
        'layer1': LAYER_WEIGHTS['layer1'],
        'layer2': LAYER_WEIGHTS['layer2'],
        'layer3': LAYER_WEIGHTS['layer3'],
        'ae': 0.2,
        'st': 0.8,
        'bank': 0.2,
    }


def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def resolve_ckpt_path(model_dir, prefix, epoch):
    preferred = os.path.join(model_dir, f'{prefix}_epoch_{epoch}.pth')
    fallback = os.path.join(model_dir, f'{prefix}_last.pth')
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f'Checkpoint not found: {preferred} or {fallback}')


def load_memory_bank(model_dir, device):
    bank_path = os.path.join(model_dir, 'memory_bank_layer2.npy')
    if not os.path.exists(bank_path):
        return None
    bank = np.load(bank_path)
    return torch.from_numpy(bank).to(device)


def load_feature_stats(model_dir, device):
    stats_path = os.path.join(model_dir, 'feature_stats.npz')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f'Missing feature stats file: {stats_path}. Please retrain model with updated train.py.')

    stats = np.load(stats_path)
    mean_std = {}
    for layer in PYRAMID_LAYERS:
        mean_std[layer] = {
            'mean': torch.from_numpy(stats[f'{layer}_mean']).to(device),
            'std': torch.from_numpy(stats[f'{layer}_std']).to(device),
        }
    return mean_std


def normalize_teacher_features(feats, mean_std):
    norm = {}
    for layer in PYRAMID_LAYERS:
        mean = mean_std[layer]['mean']
        std = mean_std[layer]['std']
        norm[layer] = (feats[layer] - mean) / (std + 1e-6)
    return norm


def compute_bank_distance_map(layer2_feat, memory_bank, chunk_size=4096):
    if memory_bank is None:
        return None

    b, c, h, w = layer2_feat.shape
    bank = memory_bank
    bank_sq_full = (bank ** 2).sum(dim=1)
    maps = []

    for n in range(b):
        query = layer2_feat[n:n + 1].permute(0, 2, 3, 1).reshape(-1, c)
        query_sq = (query ** 2).sum(dim=1, keepdim=True)

        min_dist = None
        for i in range(0, bank.shape[0], chunk_size):
            bank_chunk = bank[i:i + chunk_size]
            bank_sq = bank_sq_full[i:i + chunk_size].unsqueeze(0)
            dist_sq = torch.clamp(query_sq + bank_sq - 2.0 * (query @ bank_chunk.t()), min=0.0)
            chunk_min = torch.min(dist_sq, dim=1).values
            min_dist = chunk_min if min_dist is None else torch.minimum(min_dist, chunk_min)

        dist_map = min_dist.view(1, 1, h, w)
        maps.append(dist_map)

    dist_map = torch.cat(maps, dim=0)
    dist_map = F.interpolate(dist_map, size=(256, 256), mode='bilinear', align_corners=False)
    return dist_map


def connected_components(binary_mask):
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    ys, xs = np.where(binary_mask)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        coords = []

        while q:
            y, x = q.popleft()
            coords.append((y, x))
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and binary_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

        comp_mask = np.zeros((h, w), dtype=bool)
        ys_c, xs_c = zip(*coords)
        comp_mask[ys_c, xs_c] = True
        components.append(comp_mask)

    return components


def compute_aupro(anomaly_maps, gt_masks, max_fpr=0.3, num_thresholds=40):
    gt_binary = [(m > 0.5) for m in gt_masks]
    regions_per_image = [connected_components(mask) for mask in gt_binary]

    all_scores = np.concatenate([m.reshape(-1) for m in anomaly_maps])
    t_min, t_max = float(all_scores.min()), float(all_scores.max())
    thresholds = np.linspace(t_max, t_min, num_thresholds)

    neg_masks = [~mask for mask in gt_binary]
    total_neg = np.sum([m.sum() for m in neg_masks])
    if total_neg == 0:
        return float('nan')

    pro_points = []
    fpr_points = []

    for thr in thresholds:
        fp = 0
        region_overs = []

        for pred_map, gt_mask, regions, neg_mask in zip(anomaly_maps, gt_binary, regions_per_image, neg_masks):
            pred_bin = pred_map >= thr
            fp += np.logical_and(pred_bin, neg_mask).sum()

            if gt_mask.any() and len(regions) > 0:
                for region in regions:
                    inter = np.logical_and(pred_bin, region).sum()
                    region_area = max(1, region.sum())
                    region_overs.append(inter / region_area)

        fpr = fp / max(1, total_neg)
        pro = float(np.mean(region_overs)) if len(region_overs) > 0 else 0.0

        if fpr <= max_fpr:
            fpr_points.append(fpr)
            pro_points.append(pro)

    if len(fpr_points) < 2:
        return float('nan')

    fpr_points = np.array(fpr_points)
    pro_points = np.array(pro_points)
    order = np.argsort(fpr_points)
    fpr_points = fpr_points[order]
    pro_points = pro_points[order]

    fpr_norm = fpr_points / max_fpr
    aupro = np.trapz(pro_points, fpr_norm)
    return float(np.clip(aupro, 0.0, 1.0))


def compute_anomaly_map(teacher_feats, student_feats, ae_out_layer3):
    maps = []
    for layer in PYRAMID_LAYERS:
        map_layer = torch.mean((teacher_feats[layer] - student_feats[layer]) ** 2, dim=1, keepdim=True)
        map_layer = F.interpolate(map_layer, size=(256, 256), mode='bilinear', align_corners=False)
        maps.append(LAYER_WEIGHTS[layer] * map_layer)

    map_st = maps[0] + maps[1] + maps[2]

    map_ae = torch.mean((teacher_feats['layer3'] - ae_out_layer3) ** 2, dim=1, keepdim=True)
    map_ae = F.interpolate(map_ae, size=(256, 256), mode='bilinear', align_corners=False)

    return 0.8 * map_st + 0.2 * map_ae


def build_error_calibration(args, device, teacher, student, autoencoder, mean_std, memory_bank=None):
    calib_dataset = MVTecDataset(args.data_path, args.class_name, is_train=True, robustness_aug=False)
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=max(1, min(getattr(args, 'batch_size', 8), 8)),
        shuffle=False,
        num_workers=getattr(args, 'num_workers', 2),
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    limit_batches = max(1, int(getattr(args, 'calib_batches', 16)))
    map_size = (256, 256)
    acc = {
        'layer1': {'sum': torch.zeros((1, 1, *map_size), device=device), 'sumsq': torch.zeros((1, 1, *map_size), device=device), 'count': 0},
        'layer2': {'sum': torch.zeros((1, 1, *map_size), device=device), 'sumsq': torch.zeros((1, 1, *map_size), device=device), 'count': 0},
        'layer3': {'sum': torch.zeros((1, 1, *map_size), device=device), 'sumsq': torch.zeros((1, 1, *map_size), device=device), 'count': 0},
        'ae': {'sum': torch.zeros((1, 1, *map_size), device=device), 'sumsq': torch.zeros((1, 1, *map_size), device=device), 'count': 0},
        'bank': {'sum': torch.zeros((1, 1, *map_size), device=device), 'sumsq': torch.zeros((1, 1, *map_size), device=device), 'count': 0},
    }

    with torch.no_grad():
        for i, (img, _, _) in enumerate(calib_loader):
            if i >= limit_batches:
                break

            img = img.to(device, non_blocking=True)
            teacher_feats = teacher(img)
            teacher_feats = normalize_teacher_features(teacher_feats, mean_std)
            student_feats = student(img)
            ae_out = autoencoder(teacher_feats['layer3'])

            for layer in PYRAMID_LAYERS:
                err_map = torch.mean((teacher_feats[layer] - student_feats[layer]) ** 2, dim=1, keepdim=True)
                err_map = F.interpolate(err_map, size=(256, 256), mode='bilinear', align_corners=False)
                acc[layer]['sum'] += err_map.sum(dim=0, keepdim=True)
                acc[layer]['sumsq'] += (err_map ** 2).sum(dim=0, keepdim=True)
                acc[layer]['count'] += int(err_map.shape[0])

            ae_map = torch.mean((teacher_feats['layer3'] - ae_out) ** 2, dim=1, keepdim=True)
            ae_map = F.interpolate(ae_map, size=(256, 256), mode='bilinear', align_corners=False)
            acc['ae']['sum'] += ae_map.sum(dim=0, keepdim=True)
            acc['ae']['sumsq'] += (ae_map ** 2).sum(dim=0, keepdim=True)
            acc['ae']['count'] += int(ae_map.shape[0])

            if memory_bank is not None:
                bank_map = compute_bank_distance_map(teacher_feats['layer2'], memory_bank)
                acc['bank']['sum'] += bank_map.sum(dim=0, keepdim=True)
                acc['bank']['sumsq'] += (bank_map ** 2).sum(dim=0, keepdim=True)
                acc['bank']['count'] += int(bank_map.shape[0])

    calib = {}
    for key, value in acc.items():
        count = max(1, value['count'])
        mean = value['sum'] / count
        var = torch.clamp(value['sumsq'] / count - mean * mean, min=1e-12)
        calib[key] = {
            'mean': mean,
            'std': torch.sqrt(var),
        }
    return calib


def compute_anomaly_map_calibrated(teacher_feats, student_feats, ae_out_layer3, calib, fusion_weights):
    maps = []
    for layer in PYRAMID_LAYERS:
        map_layer = torch.mean((teacher_feats[layer] - student_feats[layer]) ** 2, dim=1, keepdim=True)
        map_layer = F.interpolate(map_layer, size=(256, 256), mode='bilinear', align_corners=False)
        mean = calib[layer]['mean']
        std = calib[layer]['std']
        map_layer = torch.relu((map_layer - mean) / (std + 1e-6))
        maps.append(fusion_weights[layer] * map_layer)

    map_st = maps[0] + maps[1] + maps[2]

    map_ae = torch.mean((teacher_feats['layer3'] - ae_out_layer3) ** 2, dim=1, keepdim=True)
    map_ae = F.interpolate(map_ae, size=(256, 256), mode='bilinear', align_corners=False)
    map_ae = torch.relu((map_ae - calib['ae']['mean']) / (calib['ae']['std'] + 1e-6))

    return fusion_weights['st'] * map_st + fusion_weights['ae'] * map_ae


def fuse_with_memory_bank(base_map, teacher_feats, memory_bank, calib, fusion_weights):
    if memory_bank is None:
        return base_map

    bank_map = compute_bank_distance_map(teacher_feats['layer2'], memory_bank)
    bank_map = torch.relu((bank_map - calib['bank']['mean']) / (calib['bank']['std'] + 1e-6))

    bank_weight = float(fusion_weights.get('bank', 0.0))
    return (1.0 - bank_weight) * base_map + bank_weight * bank_map


def evaluate(args):
    device = resolve_device(getattr(args, 'device', 'auto'))

    test_dataset = MVTecDataset(args.data_path, args.class_name, is_train=False, robustness_aug=False)
    num_workers = getattr(args, 'num_workers', 2)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    teacher = Teacher(args.pdn_size).to(device)
    student = Student(args.pdn_size).to(device)
    autoencoder = AutoEncoder(out_channels=FEATURE_CHANNELS['layer3']).to(device)

    model_dir = os.path.join(args.output_dir, 'models', args.class_name)
    teacher_path = os.path.join(model_dir, 'teacher_initial.pth')
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f'Missing teacher checkpoint: {teacher_path}')

    student_path = resolve_ckpt_path(model_dir, 'student', args.epoch)
    ae_path = resolve_ckpt_path(model_dir, 'autoencoder', args.epoch)
    mean_std = load_feature_stats(model_dir, device)
    memory_bank = load_memory_bank(model_dir, device)

    try:
        teacher.load_pretrained_teacher(teacher_path, map_location=device)
        student.load_state_dict(torch.load(student_path, map_location=device))
        autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
    except RuntimeError as exc:
        raise RuntimeError(
            'Model checkpoint is incompatible with current upgraded architecture. '
            'Please rerun training for this class with updated train.py, then evaluate again.'
        ) from exc

    teacher.eval()
    student.eval()
    autoencoder.eval()

    print('Calibrating anomaly-map statistics using train/good samples...')
    calib = build_error_calibration(args, device, teacher, student, autoencoder, mean_std, memory_bank=memory_bank)
    fusion_weights = get_fusion_weights(args)

    anomaly_maps = []
    labels = []
    gt_masks = []

    print(f'Evaluating class {args.class_name}...')
    with torch.no_grad():
        for i, (img, label, mask) in enumerate(tqdm(test_loader)):
            img = img.to(device, non_blocking=True)
            labels.append(int(label.item()))

            if mask is not None and mask.numel() > 1:
                gt_masks.append(mask.cpu().numpy()[0, 0])
            else:
                gt_masks.append(np.zeros((256, 256), dtype=np.float32))

            teacher_feats = teacher(img)
            teacher_feats = normalize_teacher_features(teacher_feats, mean_std)
            student_feats = student(img)
            ae_out = autoencoder(teacher_feats['layer3'])

            map_combined = compute_anomaly_map_calibrated(teacher_feats, student_feats, ae_out, calib, fusion_weights)
            map_combined = fuse_with_memory_bank(map_combined, teacher_feats, memory_bank, calib, fusion_weights)
            anomaly_map = map_combined.cpu().numpy()[0, 0]
            anomaly_maps.append(anomaly_map)

            if i < 3:
                os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)
                img_np = img.cpu().numpy()[0].transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_vis = np.clip(img_np * std + mean, 0.0, 1.0)

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(img_vis)
                plt.title('Input')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(anomaly_map, cmap='jet')
                plt.title('Anomaly Map')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                if int(label.item()) == 1:
                    plt.imshow(mask.cpu().numpy()[0, 0], cmap='gray')
                else:
                    plt.imshow(np.zeros((256, 256), dtype=np.float32), cmap='gray')
                plt.title('GT')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'vis', f'{args.class_name}_{i}.png'))
                plt.close()

    flat_maps = np.array(anomaly_maps).reshape(len(anomaly_maps), -1)
    image_scores = np.percentile(flat_maps, 99.0, axis=1).astype(np.float32).tolist()
    auroc = roc_auc_score(labels, image_scores)
    print(f'Image AUROC: {auroc * 100:.2f}%')

    pixel_labels = (np.array(gt_masks) > 0.5).astype(np.uint8).flatten()
    pixel_scores = np.array(anomaly_maps).flatten()
    if np.unique(pixel_labels).size > 1:
        pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)
    else:
        pixel_auroc = float('nan')
    print(f'Pixel AUROC: {pixel_auroc * 100:.2f}%')

    pixel_aupro = compute_aupro(anomaly_maps, gt_masks, max_fpr=0.3, num_thresholds=40)
    print(f'Pixel AU-PRO@30%FPR: {pixel_aupro * 100:.2f}%')

    return auroc, pixel_auroc, labels, image_scores, anomaly_maps, gt_masks


if __name__ == '__main__':
    if USE_EMBEDDED_CONFIG:
        presets = get_run_presets()
        key = f'evaluate_{EMBEDDED_CLASS}'
        if key not in presets:
            key = f'eval_{EMBEDDED_CLASS}'
        args = presets[key]
        for k, v in EMBEDDED_OVERRIDES.items():
            setattr(args, k, v)
        print('Using embedded config mode in evaluate.py')
        evaluate(args)
        raise SystemExit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DEFAULTS['data_path'])
    parser.add_argument('--class_name', type=str, default=DEFAULTS['class_name'])
    parser.add_argument('--output_dir', type=str, default=DEFAULTS['output_dir'])
    parser.add_argument('--pdn_size', type=str, default=DEFAULTS['pdn_size'])
    parser.add_argument('--epoch', type=int, default=DEFAULTS['epoch'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=DEFAULTS['device'])
    args = parser.parse_args()

    evaluate(args)
