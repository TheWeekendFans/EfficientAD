import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import Teacher


def main():
    t = Teacher()
    current = t.state_dict()
    saved = torch.load('output/models/grid/teacher_initial.pth', map_location='cpu')

    key = 'backbone.stem.0.weight'
    diff = (current[key] - saved[key]).abs().mean().item()
    print('mean_abs_diff_conv1 =', diff)


if __name__ == '__main__':
    main()
