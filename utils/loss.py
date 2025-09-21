from torch import nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.sparsity_loss_weight = 1e-10

    def forward(self, inputs, targets, valid_mask=None, sparsity_loss=False):
        if valid_mask is not None:
            inputs['rgb_coarse'] = inputs['rgb_coarse'][valid_mask]
            targets = targets[valid_mask]
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            if valid_mask is not None:
                inputs['rgb_fine'] = inputs['rgb_fine'][valid_mask]
            loss += self.loss(inputs['rgb_fine'], targets)

        if sparsity_loss:
            coarse_sparsity = inputs.get('coarse_sparsity_loss', 0)
            if hasattr(coarse_sparsity, 'sum'):
                loss += coarse_sparsity.sum() * self.sparsity_loss_weight
            else:
                loss += coarse_sparsity * self.sparsity_loss_weight

            fine_sparsity = inputs.get('fine_sparsity_loss', 0)
            if hasattr(fine_sparsity, 'sum'):
                loss += fine_sparsity.sum() * self.sparsity_loss_weight
            else:
                loss += fine_sparsity * self.sparsity_loss_weight

        return loss
