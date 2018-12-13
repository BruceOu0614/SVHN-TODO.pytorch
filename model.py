import os
import time
from typing import Tuple

import torch
import torch.nn.functional
from torch import nn, Tensor


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # TODO: CODE BEGIN
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Linear(192 * 7 * 7, 3072),
            nn.ReLU()
        )
        hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )

        self._features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )
        self._classifier = nn.Sequential(
            hidden9,
            hidden10
        )
        self._digit_length = nn.Sequential(nn.Linear(3072, 7))
        self._digit1 = nn.Sequential(nn.Linear(3072, 11))
        self._digit2 = nn.Sequential(nn.Linear(3072, 11))
        self._digit3 = nn.Sequential(nn.Linear(3072, 11))
        self._digit4 = nn.Sequential(nn.Linear(3072, 11))
        self._digit5 = nn.Sequential(nn.Linear(3072, 11))

        #raise NotImplementedError
        # TODO: CODE END

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: CODE BEGIN
        x = self._features(images)
        x = x.view(x.size(0), 192 * 7 * 7)
        x = self._classifier(x)

        logits = [self._digit_length(x), self._digit1(x), self._digit2(x), self._digit3(x), self._digit4(x), self._digit5(x)]
        return logits
        #raise NotImplementedError
        # TODO: CODE END

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        # TODO: CODE BEGIN
        labels = torch.chunk(labels, 6, dim=1)
        length = torch.squeeze(labels[0])
        dig1 = torch.squeeze(labels[1])
        dig2 = torch.squeeze(labels[2])
        dig3 = torch.squeeze(labels[3])
        dig4 = torch.squeeze(labels[4])
        dig5 = torch.squeeze(labels[5])

        length_cross_entropy = torch.nn.functional.cross_entropy(logits[0], length)
        digit1_cross_entropy = torch.nn.functional.cross_entropy(logits[1], dig1)
        digit2_cross_entropy = torch.nn.functional.cross_entropy(logits[2], dig2)
        digit3_cross_entropy = torch.nn.functional.cross_entropy(logits[3], dig3)
        digit4_cross_entropy = torch.nn.functional.cross_entropy(logits[4], dig4)
        digit5_cross_entropy = torch.nn.functional.cross_entropy(logits[5], dig5)
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
        return loss
        #raise NotImplementedError
        # TODO: CODE END

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self
