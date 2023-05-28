########################################################################
# 2. DEFINE YOUR CONVOLUTIONAL NEURAL NETWORK
########################################################################

import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # INITIALIZE LAYERS HERE
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 256, 6),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )

        self.second_layer = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        self.third_layer = nn.Sequential(
            nn.Conv2d(384, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        self.fourth_layer = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU()
        )

        self.classification = nn.Linear(4096, 10)

        # if init_weights:
        #     self._initialize_weights()

    def forward(self, x):
        # PASS IMAGE X THROUGH EACH LAYER DEFINED ABOVE
        out = self.first_layer(x)
        out = self.second_layer(out)
        out = self.third_layer(out)
        out = self.fourth_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.classification(out)
        return out

    # def _initialize_weights(self):
    #     #INITIALIZE WEIGHTS
