import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from runtime_config import get_run_presets
from evaluate import evaluate


def main():
    presets = get_run_presets()
    args = presets['evaluate_grid']
    auroc, pixel_auroc, labels, image_scores, anomaly_maps, _ = evaluate(args)

    labels = np.array(labels)
    maps = np.array(anomaly_maps)
    flat = maps.reshape(len(maps), -1)
    sorted_flat = np.sort(flat, axis=1)

    score_max = flat.max(axis=1)
    score_p999 = np.percentile(flat, 99.9, axis=1)
    score_p99 = np.percentile(flat, 99.0, axis=1)
    score_top200 = sorted_flat[:, -200:].mean(axis=1)
    score_top1000 = sorted_flat[:, -1000:].mean(axis=1)

    print('Base image AUROC (current eval):', auroc)
    print('AUROC max     :', roc_auc_score(labels, score_max))
    print('AUROC p99.9   :', roc_auc_score(labels, score_p999))
    print('AUROC p99     :', roc_auc_score(labels, score_p99))
    print('AUROC top200  :', roc_auc_score(labels, score_top200))
    print('AUROC top1000 :', roc_auc_score(labels, score_top1000))


if __name__ == '__main__':
    main()
