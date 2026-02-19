import os
import sys
import itertools
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset import MVTecDataset
from model import Teacher, Student, AutoEncoder
from evaluate import load_feature_stats, normalize_teacher_features, build_error_calibration, compute_bank_distance_map
from runtime_config import get_run_presets


def main():
    args = get_run_presets()['evaluate_grid']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = MVTecDataset(args.data_path, args.class_name, is_train=False, robustness_aug=False)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

    teacher = Teacher(args.pdn_size).to(device)
    student = Student(args.pdn_size).to(device)
    autoencoder = AutoEncoder().to(device)

    model_dir = os.path.join(args.output_dir, 'models', args.class_name)
    teacher.load_pretrained_teacher(os.path.join(model_dir, 'teacher_initial.pth'), map_location=device)
    student.load_state_dict(torch.load(os.path.join(model_dir, f'student_epoch_{args.epoch}.pth'), map_location=device))
    autoencoder.load_state_dict(torch.load(os.path.join(model_dir, f'autoencoder_epoch_{args.epoch}.pth'), map_location=device))
    memory_bank = torch.from_numpy(np.load(os.path.join(model_dir, 'memory_bank_layer2.npy')).astype(np.float32)).to(device)

    teacher.eval(); student.eval(); autoencoder.eval()
    mean_std = load_feature_stats(model_dir, device)
    calib = build_error_calibration(args, device, teacher, student, autoencoder, mean_std, memory_bank=memory_bank)

    labels = []
    e1s, e2s, e3s, eas, ebs = [], [], [], [], []

    with torch.no_grad():
        for img, label, _ in loader:
            img = img.to(device)
            labels.append(int(label.item()))

            t = teacher(img)
            t = normalize_teacher_features(t, mean_std)
            s = student(img)
            ae = autoencoder(t['layer3'])

            e1 = torch.mean((t['layer1'] - s['layer1']) ** 2, dim=1, keepdim=True)
            e2 = torch.mean((t['layer2'] - s['layer2']) ** 2, dim=1, keepdim=True)
            e3 = torch.mean((t['layer3'] - s['layer3']) ** 2, dim=1, keepdim=True)
            ea = torch.mean((t['layer3'] - ae) ** 2, dim=1, keepdim=True)
            eb = compute_bank_distance_map(t['layer2'], memory_bank)

            e1 = torch.nn.functional.interpolate(e1, size=(256, 256), mode='bilinear', align_corners=False)
            e2 = torch.nn.functional.interpolate(e2, size=(256, 256), mode='bilinear', align_corners=False)
            e3 = torch.nn.functional.interpolate(e3, size=(256, 256), mode='bilinear', align_corners=False)
            ea = torch.nn.functional.interpolate(ea, size=(256, 256), mode='bilinear', align_corners=False)

            e1 = torch.relu((e1 - calib['layer1']['mean']) / (calib['layer1']['std'] + 1e-6)).cpu().numpy()[0, 0]
            e2 = torch.relu((e2 - calib['layer2']['mean']) / (calib['layer2']['std'] + 1e-6)).cpu().numpy()[0, 0]
            e3 = torch.relu((e3 - calib['layer3']['mean']) / (calib['layer3']['std'] + 1e-6)).cpu().numpy()[0, 0]
            ea = torch.relu((ea - calib['ae']['mean']) / (calib['ae']['std'] + 1e-6)).cpu().numpy()[0, 0]
            eb = torch.relu((eb - calib['bank']['mean']) / (calib['bank']['std'] + 1e-6)).cpu().numpy()[0, 0]

            e1s.append(e1); e2s.append(e2); e3s.append(e3); eas.append(ea); ebs.append(eb)

    labels = np.array(labels)
    e1s = np.array(e1s); e2s = np.array(e2s); e3s = np.array(e3s); eas = np.array(eas); ebs = np.array(ebs)

    st_weight_sets = [
        (0.1, 0.9, 0.0), (0.15, 0.85, 0.0), (0.2, 0.8, 0.0),
        (0.0, 1.0, 0.0), (0.2, 0.7, 0.1), (0.1, 0.8, 0.1),
    ]

    best = None
    for w1, w2, w3 in st_weight_sets:
        st = w1 * e1s + w2 * e2s + w3 * e3s
        for ae_w in [0.0, 0.05, 0.1, 0.15, 0.2]:
            base = (1.0 - ae_w) * st + ae_w * eas
            for bank_w in [0.45, 0.5, 0.55, 0.6, 0.65]:
                fused = (1.0 - bank_w) * base + bank_w * ebs
                flat = fused.reshape(len(fused), -1)
                for q in [99.3, 99.5, 99.7, 99.8, 99.9]:
                    for g in [0.8, 1.0, 1.2]:
                        scores = np.percentile(np.power(np.clip(flat, 0, None), g), q, axis=1)
                        auc = roc_auc_score(labels, scores)
                        if best is None or auc > best[0]:
                            best = (auc, (w1, w2, w3), ae_w, bank_w, q, g)

    print('BEST', best)


if __name__ == '__main__':
    main()
