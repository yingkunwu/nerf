from torch import nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets, valid_mask=None):
        if valid_mask is not None:
            inputs['rgb_coarse'] = inputs['rgb_coarse'][valid_mask]
            targets = targets[valid_mask]
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            if valid_mask is not None:
                inputs['rgb_fine'] = inputs['rgb_fine'][valid_mask]
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss
