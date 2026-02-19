import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch.utils.data import DataLoader
from dataset import MVTecDataset
from model import Teacher, Student, AutoEncoder
from evaluate import load_feature_stats, normalize_teacher_features, compute_anomaly_map
from runtime_config import get_run_presets


def q(arr):
    return {
        'min': float(np.min(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
    }


def main():
    args = get_run_presets()['evaluate_grid']
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))

    test_dataset = MVTecDataset(args.data_path, args.class_name, is_train=False, robustness_aug=False)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=='cuda'))

    teacher = Teacher(args.pdn_size).to(device)
    student = Student(args.pdn_size).to(device)
    autoencoder = AutoEncoder().to(device)

    model_dir = os.path.join(args.output_dir, 'models', args.class_name)
    teacher.load_pretrained_teacher(os.path.join(model_dir, 'teacher_initial.pth'), map_location=device)
    student.load_state_dict(torch.load(os.path.join(model_dir, f'student_epoch_{args.epoch}.pth'), map_location=device))
    autoencoder.load_state_dict(torch.load(os.path.join(model_dir, f'autoencoder_epoch_{args.epoch}.pth'), map_location=device))

    mean_std = load_feature_stats(model_dir, device)

    teacher.eval(); student.eval(); autoencoder.eval()

    labels = []
    score_max = []
    score_p99 = []
    layer3_abs_mean = []
    layer3_abs_std = []

    with torch.no_grad():
        for img, label, _ in loader:
            img = img.to(device)
            labels.append(int(label.item()))

            t = teacher(img)
            t = normalize_teacher_features(t, mean_std)
            s = student(img)
            ae = autoencoder(t['layer3'])
            m = compute_anomaly_map(t, s, ae).cpu().numpy()[0, 0]

            flat = m.reshape(-1)
            score_max.append(float(np.max(flat)))
            score_p99.append(float(np.percentile(flat, 99.0)))

            l3 = t['layer3'].detach().cpu().numpy().reshape(-1)
            layer3_abs_mean.append(float(np.mean(np.abs(l3))))
            layer3_abs_std.append(float(np.std(l3)))

    labels = np.array(labels)
    score_max = np.array(score_max)
    score_p99 = np.array(score_p99)

    normal = labels == 0
    anom = labels == 1

    print('Counts:', int(normal.sum()), int(anom.sum()))
    print('MAX normal :', q(score_max[normal]))
    print('MAX anom   :', q(score_max[anom]))
    print('P99 normal :', q(score_p99[normal]))
    print('P99 anom   :', q(score_p99[anom]))
    print('L3 |abs| mean:', q(np.array(layer3_abs_mean)))
    print('L3 std      :', q(np.array(layer3_abs_std)))


if __name__ == '__main__':
    main()
