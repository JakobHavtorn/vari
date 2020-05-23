import torch.nn as nn
import torch.functional as F


class SpatialDropout2D(nn.Dropout):
    def forward(self, x):
        x = x.permute(0, 2, 1)   # convert to [batch, channels, time]
        x = F.dropout2d(input, self.p, self.training, self.inplace)
        x = x.permute(0, 2, 1)   # back to [batch, time, channels]
        return x
        
