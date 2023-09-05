import torch
from torch import nn
from torch.nn import functional as F
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, act="relu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leaky":
            # self.act = nn.LeakyReLU(inplace=True)
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "linear":
            self.act = None
        else:
            raise Exception("Activation not available")

        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        if self.act is None:
            return out
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, norm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if norm:
            self.conv_block = nn.Sequential(
                                nn.Conv2d(cin, cout, kernel_size, stride, padding),
                                nn.InstanceNorm2d(cout, affine=True)
                                )
        else:
            self.conv_block = nn.Sequential(
                                nn.Conv2d(cin, cout, kernel_size, stride, padding),
                                )
        # self.act = nn.LeakyReLU(0.01, inplace=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
    
class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, act="leaky", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        # self.act = nn.SiLU(inplace=True)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leaky":
            # self.act = nn.LeakyReLU(inplace=True)
            self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)