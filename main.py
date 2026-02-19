import argparse
from runtime_config import DEFAULTS, get_run_presets


USE_EMBEDDED_CONFIG = True

# 直接运行 main.py 时，修改这两个值即可切换任务
EMBEDDED_MODE = 'evaluate'       # train / evaluate / analysis
EMBEDDED_CLASS = 'grid'       # grid / transistor

# 可选：覆盖默认配置，不填则使用 runtime_config.py 里的 DEFAULTS
EMBEDDED_OVERRIDES = {
    # 'data_path': r'F:\\lhd\\2',
    # 'device': 'cuda',
    # 'epochs': 100,
}


def run_with_args(args):
    if args.mode == 'train':
        from train import train
        print(f"Starting Training for {args.class_name}...")
        train(args)
    elif args.mode == 'evaluate':
        from evaluate import evaluate
        print(f"Starting Evaluation for {args.class_name}...")
        evaluate(args)
    elif args.mode == 'analysis':
        from evaluate import evaluate
        from production_analysis import production_analysis
        print(f"Starting Production Analysis for {args.class_name}...")
        _, _, labels, scores, _, _ = evaluate(args)
        production_analysis(args, scores, labels)

def main():
    if USE_EMBEDDED_CONFIG:
        presets = get_run_presets()
        mode_alias = {
            'train': 'train',
            'evaluate': 'evaluate',
            'eval': 'evaluate',
            'analysis': 'analysis',
        }

        normalized_mode = mode_alias.get(EMBEDDED_MODE, EMBEDDED_MODE)
        preset_key = f"{normalized_mode}_{EMBEDDED_CLASS}"
        if preset_key not in presets:
            raise ValueError(f"Unknown embedded preset: {preset_key}")

        args = presets[preset_key]
        for key, value in EMBEDDED_OVERRIDES.items():
            setattr(args, key, value)

        print("Using embedded config mode. To switch task, edit EMBEDDED_MODE / EMBEDDED_CLASS in main.py")
        run_with_args(args)
        return

    parser = argparse.ArgumentParser(description="Semiconductor Wafer Defect Detection System")
    parser.add_argument('--mode', type=str, default=DEFAULTS['mode'], choices=['train', 'evaluate', 'analysis'], help='Operation mode')
    parser.add_argument('--data_path', type=str, default=DEFAULTS['data_path'], help='Path to dataset root')
    parser.add_argument('--class_name', type=str, default=DEFAULTS['class_name'], help='Dataset category (grid or transistor)')
    parser.add_argument('--output_dir', type=str, default=DEFAULTS['output_dir'], help='Output directory')
    parser.add_argument('--pdn_size', type=str, default=DEFAULTS['pdn_size'], choices=['small', 'medium'])
    parser.add_argument('--teacher_path', type=str, default=DEFAULTS['teacher_path'], help='Path to pretrained teacher weights')
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'], help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'], help='Number of epochs')
    parser.add_argument('--epoch', type=int, default=DEFAULTS['epoch'], help='Checkpoint epoch used for evaluate/analysis')
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'], help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=DEFAULTS['num_workers'], help='Dataloader workers')
    parser.add_argument('--device', type=str, default=DEFAULTS['device'], help='cuda / cuda:0 / cpu / auto')
    parser.add_argument('--amp', action='store_true', default=DEFAULTS['amp'], help='Use mixed precision training on CUDA')
    parser.add_argument('--save_every', type=int, default=DEFAULTS['save_every'], help='Save checkpoint every N epochs')
    parser.add_argument('--adaptive_k', type=float, default=DEFAULTS['adaptive_k'], help='k for adaptive threshold: mu + k*sigma')
    parser.add_argument('--target_overkill', type=float, default=DEFAULTS['target_overkill'], help='target overkill for adaptive threshold quantile fitting')
    
    args = parser.parse_args()

    run_with_args(args)

if __name__ == '__main__':
    main()
