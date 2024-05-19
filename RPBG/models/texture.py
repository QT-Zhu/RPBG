import torch
import torch.nn as nn


class PointTexture(nn.Module):
    def __init__(self, num_channels, size, activation='none', checkpoint=None, init_method='zeros', reg_weight=0.):
        super().__init__()

        assert isinstance(size, int), 'size must be int'

        shape = 1, num_channels, size

        if checkpoint:
            self.texture_ = torch.load(checkpoint, map_location='cpu')['texture'].texture_
        else:
            if init_method == 'rand':
                texture = torch.rand(shape)
            elif init_method == 'zeros':
                texture = torch.zeros(shape)
            else:
                raise ValueError(init_method)
            self.texture_ = nn.Parameter(texture.float())

        self.activation = activation
        self.reg_weight = reg_weight

    def null_grad(self):
        self.texture_.grad = None

    def reg_loss(self):
        return self.reg_weight * torch.mean(torch.pow(self.texture_, 2))

    def forward(self, inputs):
        if isinstance(inputs, dict):
            ids = None
            for f, x in inputs.items():
                if 'uv' in f:
                    ids = x[:, 0].long()
            assert ids is not None, 'Input format does not have uv'
        else:
            if len(inputs.shape)==4: # Bx3xHxW
                ids = inputs[:, 0] # BxHxW
            else:
                ids = inputs
                
        sh = ids.shape
        n_pts = self.texture_.shape[-1]

        assert ids.max() < n_pts, f"{ids.max()=} is out of the range of 0-{n_pts-1}, this is probably due to precision issues"
        ind = ids.contiguous().view(-1).long().cuda()
        texture = self.texture_.permute(1, 0, 2) # Cx1xN
        texture = texture.expand(texture.shape[0], sh[0], texture.shape[2]) # CxBxN
        texture = texture.contiguous().view(texture.shape[0], -1) # CxB*N
        sample = torch.index_select(texture, 1, ind) # CxB*H*W
        sample = sample.contiguous().view(sample.shape[0], sh[0], sh[1], sh[2]) # CxBxHxW
        sample = sample.permute(1, 0, 2, 3) # BxCxHxW

        if self.activation == 'sigmoid':
            return torch.sigmoid(sample)
        elif self.activation == 'tanh':
            return torch.tanh(sample)
        return sample
