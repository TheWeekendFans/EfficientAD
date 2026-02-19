from types import SimpleNamespace


DEFAULTS = {
    'mode': 'train',
    'data_path': r'F:\lhd\2',
    'class_name': 'grid',
    'output_dir': 'output',
    'pdn_size': 'small',
    'teacher_path': None,
    'batch_size': 8,
    'epochs': 100,
    'epoch': 100,
    'lr': 1e-4,
    'num_workers': 8,
    'device': 'cuda',
    'amp': True,
    'save_every': 10,
    'hard_keep_ratio': 0.1,
    'sep_weight': 0.3,
    'sep_margin': 0.25,
    'bank_size': 20000,
    'calib_batches': 16,
    'adaptive_k': 3.0,
    'target_overkill': 0.05,
    'near_zero_escape': 0.02,
    'relaxed_escape': 0.05,
    'relaxed_overkill': 0.10,
}


def build_args(**overrides):
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def get_run_presets():
    presets = {
        'train_grid': build_args(mode='train', class_name='grid', device='cuda'),
        'train_transistor': build_args(mode='train', class_name='transistor', device='cuda'),
        'eval_grid': build_args(mode='evaluate', class_name='grid', device='cuda', num_workers=4),
        'eval_transistor': build_args(mode='evaluate', class_name='transistor', device='cuda', num_workers=4),
        'analysis_grid': build_args(mode='analysis', class_name='grid', device='cuda', num_workers=4),
        'analysis_transistor': build_args(mode='analysis', class_name='transistor', device='cuda', num_workers=4),
    }

    presets['evaluate_grid'] = presets['eval_grid']
    presets['evaluate_transistor'] = presets['eval_transistor']
    return presets
