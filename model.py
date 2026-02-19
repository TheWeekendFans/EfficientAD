import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


FEATURE_CHANNELS = {
    'layer1': 64,
    'layer2': 128,
    'layer3': 256,
}


class ResNet18Pyramid(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return {
            'layer1': f1,
            'layer2': f2,
            'layer3': f3,
        }


class Teacher(nn.Module):
    def __init__(self, size='small', out_channels=None, padding=False):
        super().__init__()
        try:
            self.backbone = ResNet18Pyramid(pretrained=True)
            print('Teacher backbone: pretrained ResNet18 loaded.')
        except Exception:
            self.backbone = ResNet18Pyramid(pretrained=False)
            print('Teacher backbone: pretrained weights unavailable, using random init.')

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

    def load_pretrained_teacher(self, path, map_location=None):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state, strict=True)
        self.eval()


class Student(nn.Module):
    def __init__(self, size='small', out_channels=None, padding=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.proj1 = nn.Conv2d(32, FEATURE_CHANNELS['layer1'], kernel_size=1)
        self.proj2 = nn.Conv2d(64, FEATURE_CHANNELS['layer2'], kernel_size=1)
        self.proj3 = nn.Conv2d(128, FEATURE_CHANNELS['layer3'], kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        return {
            'layer1': self.proj1(f1),
            'layer2': self.proj2(f2),
            'layer3': self.proj3(f3),
        }


class AutoEncoder(nn.Module):
    def __init__(self, out_channels=FEATURE_CHANNELS['layer3'], padding=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(out_channels, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
