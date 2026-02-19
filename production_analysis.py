import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset
from evaluate import (
    evaluate,
    resolve_ckpt_path,
    resolve_device,
    load_feature_stats,
    normalize_teacher_features,
    load_memory_bank,
    build_error_calibration,
    get_fusion_weights,
    compute_anomaly_map_calibrated,
    fuse_with_memory_bank,
)
from model import Teacher, Student, AutoEncoder, FEATURE_CHANNELS
from runtime_config import DEFAULTS, get_run_presets


USE_EMBEDDED_CONFIG = True
EMBEDDED_CLASS = 'grid'
EMBEDDED_OVERRIDES = {
    # 'epoch': 100,
    # 'adaptive_k': 3.0,
    # 'target_overkill': 0.05,
}


def calculate_metrics(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(np.uint8)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    escape_rate = fn / (tp + fn + 1e-10)
    overkill_rate = fp / (tn + fp + 1e-10)
    return escape_rate, overkill_rate


def production_analysis(args, model_output_scores, model_output_labels):
    os.makedirs(args.output_dir, exist_ok=True)

    y_true = np.array(model_output_labels)
    y_scores = np.array(model_output_scores)

    thresholds = np.linspace(float(np.min(y_scores)), float(np.max(y_scores)), 2000)
    escape_rates = []
    overkill_rates = []

    for thresh in thresholds:
        er, ok = calculate_metrics(y_true, y_scores, thresh)
        escape_rates.append(er)
        overkill_rates.append(ok)

    defect_scores = y_scores[y_true == 1]
    if defect_scores.size == 0:
        raise ValueError('No defect samples found in test labels; cannot compute Zero Escape threshold.')

    zero_escape_thresh = float(np.min(defect_scores) - 1e-8)
    er_zero, ok_zero = calculate_metrics(y_true, y_scores, zero_escape_thresh)

    x_ok = np.array(overkill_rates) * 100.0
    y_er = np.array(escape_rates) * 100.0

    plt.figure(figsize=(7, 5))
    plt.plot(x_ok, y_er, linewidth=2.2, color='#1f77b4', label='Trade-off Curve')
    plt.xlabel('Overkill Rate (%)')
    plt.ylabel('Escape Rate (%)')
    plt.title(f'Trade-off: {args.class_name}')
    plt.grid(True, alpha=0.3)

    plt.scatter([ok_zero * 100.0], [er_zero * 100.0], color='#d62728', s=40, label='Zero Escape')

    near_zero_escape = float(getattr(args, 'near_zero_escape', 0.02))
    valid_idx = np.where(np.array(escape_rates) <= near_zero_escape)[0]
    near_zero_result = None
    if valid_idx.size > 0:
        best_idx = valid_idx[np.argmin(np.array(overkill_rates)[valid_idx])]
        near_zero_result = (best_idx, escape_rates[best_idx], overkill_rates[best_idx], thresholds[best_idx])
        plt.scatter([overkill_rates[best_idx] * 100.0], [escape_rates[best_idx] * 100.0], color='#ff7f0e', s=40, label='Near Zero Escape')

    target_ok_low = 0.03
    target_ok_high = float(getattr(args, 'target_overkill', 0.05))
    ok_band_idx = np.where((np.array(overkill_rates) >= target_ok_low) & (np.array(overkill_rates) <= target_ok_high))[0]
    ok_band_result = None
    if ok_band_idx.size > 0:
        best_ok_idx = ok_band_idx[np.argmin(np.array(escape_rates)[ok_band_idx])]
        ok_band_result = (best_ok_idx, escape_rates[best_ok_idx], overkill_rates[best_ok_idx], thresholds[best_ok_idx])
        plt.scatter([overkill_rates[best_ok_idx] * 100.0], [escape_rates[best_ok_idx] * 100.0], color='#2ca02c', s=45, label='3-5% Overkill Band Best')

    relaxed_escape = float(getattr(args, 'relaxed_escape', 0.05))
    relaxed_overkill = float(getattr(args, 'relaxed_overkill', 0.10))
    relaxed_idx = np.where((np.array(escape_rates) <= relaxed_escape) & (np.array(overkill_rates) <= relaxed_overkill))[0]
    relaxed_result = None
    if relaxed_idx.size > 0:
        best_relaxed_idx = relaxed_idx[np.argmin(np.array(escape_rates)[relaxed_idx] + np.array(overkill_rates)[relaxed_idx])]
        relaxed_result = (
            best_relaxed_idx,
            escape_rates[best_relaxed_idx],
            overkill_rates[best_relaxed_idx],
            thresholds[best_relaxed_idx],
        )
        plt.scatter([overkill_rates[best_relaxed_idx] * 100.0], [escape_rates[best_relaxed_idx] * 100.0], color='#9467bd', s=45, label='Relaxed Target Best')

    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'tradeoff_curve_{args.class_name}.png'), dpi=220)
    plt.close()

    print(f'Zero Escape Threshold: {zero_escape_thresh:.4f}')
    print(f'At Zero Escape Rate -> Overkill Rate: {ok_zero * 100:.2f}%')

    if near_zero_result is not None:
        best_idx, _, _, _ = near_zero_result
        print(
            f'Closest Practical (Escape<={near_zero_escape * 100:.2f}%) -> '
            f'Escape: {escape_rates[best_idx] * 100:.2f}%, Overkill: {overkill_rates[best_idx] * 100:.2f}%, '
            f'Threshold: {thresholds[best_idx]:.4f}'
        )

    if ok_band_result is not None:
        best_ok_idx, _, _, _ = ok_band_result
        print(
            f'Overkill {target_ok_low * 100:.1f}-{target_ok_high * 100:.1f}% Band Best -> '
            f'Escape: {escape_rates[best_ok_idx] * 100:.2f}%, Overkill: {overkill_rates[best_ok_idx] * 100:.2f}%, '
            f'Threshold: {thresholds[best_ok_idx]:.4f}'
        )

    if relaxed_result is not None:
        best_relaxed_idx, _, _, _ = relaxed_result
        print(
            f'Relaxed Target (Escape<={relaxed_escape * 100:.1f}%, Overkill<={relaxed_overkill * 100:.1f}%) -> '
            f'Escape: {escape_rates[best_relaxed_idx] * 100:.2f}%, '
            f'Overkill: {overkill_rates[best_relaxed_idx] * 100:.2f}%, '
            f'Threshold: {thresholds[best_relaxed_idx]:.4f}'
        )
    else:
        print(
            f'Relaxed Target (Escape<={relaxed_escape * 100:.1f}%, Overkill<={relaxed_overkill * 100:.1f}%) not found.'
        )

    print('Fitting Adaptive Threshold...')
    adaptive_thresh, est_overkill = fit_adaptive_threshold(args)
    print(f'Adaptive Threshold: {adaptive_thresh:.4f} (estimated train-good overkill {est_overkill * 100:.2f}%)')

    er_adapt, ok_adapt = calculate_metrics(y_true, y_scores, adaptive_thresh)
    print(f'Adaptive Threshold Results -> Escape Rate: {er_adapt * 100:.2f}%, Overkill Rate: {ok_adapt * 100:.2f}%')

    summary_path = os.path.join(args.output_dir, f'production_summary_{args.class_name}.txt')
    lines = [
        f'class={args.class_name}',
        f'zero_escape_threshold={zero_escape_thresh:.6f}',
        f'zero_escape_overkill={ok_zero * 100:.4f}%',
        f'adaptive_threshold={adaptive_thresh:.6f}',
        f'adaptive_escape={er_adapt * 100:.4f}%',
        f'adaptive_overkill={ok_adapt * 100:.4f}%',
    ]
    if near_zero_result is not None:
        _, er, ok, th = near_zero_result
        lines.append(f'near_zero_escape_threshold={th:.6f}')
        lines.append(f'near_zero_escape={er * 100:.4f}%')
        lines.append(f'near_zero_overkill={ok * 100:.4f}%')
    if ok_band_result is not None:
        _, er, ok, th = ok_band_result
        lines.append(f'ok_band_threshold={th:.6f}')
        lines.append(f'ok_band_escape={er * 100:.4f}%')
        lines.append(f'ok_band_overkill={ok * 100:.4f}%')
    if relaxed_result is not None:
        _, er, ok, th = relaxed_result
        lines.append(f'relaxed_threshold={th:.6f}')
        lines.append(f'relaxed_escape={er * 100:.4f}%')
        lines.append(f'relaxed_overkill={ok * 100:.4f}%')
    else:
        lines.append('relaxed_threshold=NOT_FOUND')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Production summary saved to: {summary_path}')

    return zero_escape_thresh, ok_zero


def fit_adaptive_threshold(args):
    device = resolve_device(getattr(args, 'device', 'auto'))

    train_dataset = MVTecDataset(args.data_path, args.class_name, is_train=True, robustness_aug=True)
    num_workers = getattr(args, 'num_workers', 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    model_dir = os.path.join(args.output_dir, 'models', args.class_name)
    teacher = Teacher(args.pdn_size).to(device)
    student = Student(args.pdn_size).to(device)
    autoencoder = AutoEncoder(out_channels=FEATURE_CHANNELS['layer3']).to(device)

    teacher_path = os.path.join(model_dir, 'teacher_initial.pth')
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
            'Please rerun training for this class with updated train.py, then analysis again.'
        ) from exc

    teacher.eval()
    student.eval()
    autoencoder.eval()

    calib = build_error_calibration(args, device, teacher, student, autoencoder, mean_std, memory_bank=memory_bank)
    fusion_weights = get_fusion_weights(args)

    good_scores = []
    with torch.no_grad():
        for img, _, _ in tqdm(train_loader, desc='Fitting Threshold'):
            img = img.to(device, non_blocking=True)

            teacher_feats = teacher(img)
            teacher_feats = normalize_teacher_features(teacher_feats, mean_std)
            student_feats = student(img)
            ae_out = autoencoder(teacher_feats['layer3'])

            map_combined = compute_anomaly_map_calibrated(teacher_feats, student_feats, ae_out, calib, fusion_weights)
            map_combined = fuse_with_memory_bank(map_combined, teacher_feats, memory_bank, calib, fusion_weights)
            score = float(np.percentile(map_combined.detach().cpu().numpy()[0, 0], 99.0))
            good_scores.append(score)

    good_scores = np.array(good_scores, dtype=np.float32)
    target_overkill = float(getattr(args, 'target_overkill', 0.05))
    target_overkill = float(np.clip(target_overkill, 1e-4, 0.5))

    quantile = 1.0 - target_overkill
    q_thresh = float(np.quantile(good_scores, quantile))

    mu = float(np.mean(good_scores))
    sigma = float(np.std(good_scores))
    k = float(getattr(args, 'adaptive_k', 3.0))
    ksigma_thresh = mu + k * sigma

    adaptive_thresh = min(q_thresh, ksigma_thresh)
    est_overkill = float(np.mean(good_scores > adaptive_thresh))
    return adaptive_thresh, est_overkill


if __name__ == '__main__':
    if USE_EMBEDDED_CONFIG:
        presets = get_run_presets()
        args = presets[f'analysis_{EMBEDDED_CLASS}']
        for key, value in EMBEDDED_OVERRIDES.items():
            setattr(args, key, value)
        print('Using embedded config mode in production_analysis.py')
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path', type=str, default=DEFAULTS['data_path'])
        parser.add_argument('--class_name', type=str, default=DEFAULTS['class_name'])
        parser.add_argument('--output_dir', type=str, default=DEFAULTS['output_dir'])
        parser.add_argument('--pdn_size', type=str, default=DEFAULTS['pdn_size'])
        parser.add_argument('--epoch', type=int, default=DEFAULTS['epoch'])
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--device', type=str, default=DEFAULTS['device'])
        parser.add_argument('--adaptive_k', type=float, default=DEFAULTS['adaptive_k'])
        parser.add_argument('--target_overkill', type=float, default=DEFAULTS.get('target_overkill', 0.05))
        parser.add_argument('--near_zero_escape', type=float, default=DEFAULTS.get('near_zero_escape', 0.02))
        parser.add_argument('--relaxed_escape', type=float, default=DEFAULTS.get('relaxed_escape', 0.05))
        parser.add_argument('--relaxed_overkill', type=float, default=DEFAULTS.get('relaxed_overkill', 0.10))
        args = parser.parse_args()

    _, _, labels, scores, _, _ = evaluate(args)
    production_analysis(args, scores, labels)
