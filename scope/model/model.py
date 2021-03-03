import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch_3d import EfficientNet3D
from torchsummary import summary
from torchvision import models

from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class EfficientNet3DModel(BaseModel):
    def __init__(self, num_classes=6, in_channels=4, drop_connect_rate=0.2):
        super().__init__()

        self.model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': num_classes,
                                                                                  'drop_connect_rate': drop_connect_rate}, in_channels=in_channels)

    def forward(self, x):
        return self.model(x)

    def summarize(self, input_size):
        summary(self.model, input_size=input_size)


class Resnet3D_18L(BaseModel):
    def __init__(self, num_classes=6, in_channels=4):
        super().__init__()

        self.model = models.video.r3d_18(pretrained=True, progress=True)

        self.model.stem[0] = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

    def summarize(self, input_size):
        summary(self.model, input_size=input_size)
