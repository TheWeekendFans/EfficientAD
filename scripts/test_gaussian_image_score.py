import os
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset import MVTecDataset
from model import Teacher, Student, AutoEncoder
from evaluate import load_feature_stats, normalize_teacher_features, build_error_calibration
from runtime_config import get_run_presets


def extract_map_components(args, device, teacher, student, autoencoder, mean_std, calib, dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    labels = []
    feats = []

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

            e1 = torch.nn.functional.interpolate(e1, size=(256, 256), mode='bilinear', align_corners=False)
            e2 = torch.nn.functional.interpolate(e2, size=(256, 256), mode='bilinear', align_corners=False)
            e3 = torch.nn.functional.interpolate(e3, size=(256, 256), mode='bilinear', align_corners=False)
            ea = torch.nn.functional.interpolate(ea, size=(256, 256), mode='bilinear', align_corners=False)

            e1 = torch.relu((e1 - calib['layer1']['mean']) / (calib['layer1']['std'] + 1e-6)).cpu().numpy()[0, 0]
            e2 = torch.relu((e2 - calib['layer2']['mean']) / (calib['layer2']['std'] + 1e-6)).cpu().numpy()[0, 0]
            e3 = torch.relu((e3 - calib['layer3']['mean']) / (calib['layer3']['std'] + 1e-6)).cpu().numpy()[0, 0]
            ea = torch.relu((ea - calib['ae']['mean']) / (calib['ae']['std'] + 1e-6)).cpu().numpy()[0, 0]

            f = [
                np.percentile(e1, 99.0), np.percentile(e1, 99.5),
                np.percentile(e2, 99.0), np.percentile(e2, 99.5),
                np.percentile(e3, 99.0), np.percentile(e3, 99.5),
                np.percentile(ea, 99.0), np.percentile(ea, 99.5),
            ]
            feats.append(f)

    return np.array(labels), np.array(feats, dtype=np.float64)


def main():
    args = get_run_presets()['evaluate_grid']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher = Teacher(args.pdn_size).to(device)
    student = Student(args.pdn_size).to(device)
    autoencoder = AutoEncoder().to(device)

    model_dir = os.path.join(args.output_dir, 'models', args.class_name)
    teacher.load_pretrained_teacher(os.path.join(model_dir, 'teacher_initial.pth'), map_location=device)
    student.load_state_dict(torch.load(os.path.join(model_dir, f'student_epoch_{args.epoch}.pth'), map_location=device))
    autoencoder.load_state_dict(torch.load(os.path.join(model_dir, f'autoencoder_epoch_{args.epoch}.pth'), map_location=device))

    teacher.eval(); student.eval(); autoencoder.eval()

    mean_std = load_feature_stats(model_dir, device)
    calib = build_error_calibration(args, device, teacher, student, autoencoder, mean_std)

    train_dataset = MVTecDataset(args.data_path, args.class_name, is_train=True, robustness_aug=False)
    test_dataset = MVTecDataset(args.data_path, args.class_name, is_train=False, robustness_aug=False)

    _, train_feats = extract_map_components(args, device, teacher, student, autoencoder, mean_std, calib, train_dataset)
    test_labels, test_feats = extract_map_components(args, device, teacher, student, autoencoder, mean_std, calib, test_dataset)

    mu = train_feats.mean(axis=0)
    cov = np.cov(train_feats.T) + np.eye(train_feats.shape[1]) * 1e-6
    inv_cov = np.linalg.inv(cov)

    dists = []
    for x in test_feats:
        d = x - mu
        dist = float(np.sqrt(np.maximum(0.0, d @ inv_cov @ d.T)))
        dists.append(dist)

    auroc = roc_auc_score(test_labels, np.array(dists))
    print('Gaussian feature AUROC:', auroc)


if __name__ == '__main__':
    main()
