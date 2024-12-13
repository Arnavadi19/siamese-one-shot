import torch
import torch.nn as nn

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1: 1@105x105 → 64@96x96
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48

            # Layer 2: 64@48x48 → 128@42x42
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21

            # Layer 3: 128@21x21 → 128@18x18
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2), # 128@9*9

            # Layer 4: 128@9x9 → 256@6x6
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   # 256@6*6
        )
        self.liner = nn.Sequential(
            nn.Linear(9216, 4096), 
            nn.Sigmoid()  # Flatten 256@6x6 = 9216 → 4096, sigmoid normalizes to [0, 1]
        )
        self.out = nn.Linear(4096, 1) # Output

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out

