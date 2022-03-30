import torch
import torch.nn as nn
import torch.nn.functional as F


class BCNet(nn.Module):
    def __init__(self, CPoint, measure, cmd, output_size, args):
        super(BCNet, self).__init__()
        self.cp_layer1 = nn.Linear(CPoint, 128)
        self.cp_layer1.weight.data.normal_(0, 1)
        self.cp_layer2 = nn.Linear(128, 64)
        self.cp_layer2.weight.data.normal_(0, 1)

        self.mea_layer = nn.Linear(measure, 32)
        self.mea_layer.weight.data.normal_(0, 1)

        self.cmd_layer = nn.Linear(cmd, 32)
        self.cmd_layer.weight.data.normal_(0, 1)

        # 128 = 64 + 32 + 32
        self.layer_out_1 = nn.Linear(128, 64)
        self.layer_out_1.weight.data.normal_(0, 1)
        self.layer_out_2 = nn.Linear(64, output_size)
        self.layer_out_2.weight.data.normal_(0, 1)

    def forward(self, contralPoint, measurement, command):
        cp = F.relu(self.cp_layer1(contralPoint))
        cp = F.relu(self.cp_layer2(cp))

        measure = F.relu(self.mea_layer(measurement))
        cmd = F.relu(self.cmd_layer(command))
        
        x = torch.cat([cp, measure, cmd], dim=1)
        x = F.relu(self.layer_out_1(x))
        x = self.layer_out_2(x)
        return x