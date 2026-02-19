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
from evaluate import (
    load_feature_stats,
    normalize_teacher_features,
    build_error_calibration,
    compute_bank_distance_map,
)
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
    l1_maps, l2_maps, l3_maps, ae_maps, bank_maps = [], [], [], [], []

    with torch.no_grad():
        for img, label, _ in loader:
            img = img.to(device, non_blocking=True)
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

            l1_maps.append(e1)
            l2_maps.append(e2)
            l3_maps.append(e3)
            ae_maps.append(ea)
            bank_maps.append(eb)

    labels = np.array(labels)
    l1_maps = np.array(l1_maps)
    l2_maps = np.array(l2_maps)
    l3_maps = np.array(l3_maps)
    ae_maps = np.array(ae_maps)
    bank_maps = np.array(bank_maps)

    candidates = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    quantiles = [98.0, 99.0, 99.3, 99.5, 99.7, 99.9]

    best = None
    for bank_w in [0.2, 0.3, 0.4, 0.5, 0.6]:
        base_scale = 1.0 - bank_w
        for ae_w in [0.0, 0.1, 0.2, 0.3]:
            st_w = 1.0 - ae_w
            for w1, w2, w3 in itertools.product(candidates, candidates, candidates):
                s = w1 + w2 + w3
                if s == 0:
                    continue
                w1n, w2n, w3n = w1 / s, w2 / s, w3 / s
                st_map = w1n * l1_maps + w2n * l2_maps + w3n * l3_maps
                base_map = st_w * st_map + ae_w * ae_maps
                m = base_scale * base_map + bank_w * bank_maps
                flat = m.reshape(len(m), -1)

                for q in quantiles:
                    scores = np.percentile(flat, q, axis=1)
                    auroc = roc_auc_score(labels, scores)
                    if best is None or auroc > best[0]:
                        best = (auroc, bank_w, ae_w, w1n, w2n, w3n, q)

    print('BEST', best)


if __name__ == '__main__':
    main()
