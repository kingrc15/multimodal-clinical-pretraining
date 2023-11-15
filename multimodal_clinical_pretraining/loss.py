"""
    implementation of other two-way contrastive losses
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CLIP_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        learnable_temp=False,
    ):
        super(CLIP_Loss, self).__init__()
        self.world_size = world_size
        if learnable_temp:
            self.temperature = nn.Parameter(torch.ones([]) * temperature)
        else:
            self.temperature = torch.ones([]) * temperature

    def forward(self, image_features, text_features):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        self.temperature.data = torch.clamp(self.temperature, min=0.01)
        sim = torch.einsum("i d, j d -> i j", text_features, image_features)
        labels = torch.arange(image_features.shape[0], device=image_features.device)
        total_loss = (
            F.cross_entropy(sim / self.temperature, labels)
            + F.cross_entropy(sim.t() / self.temperature, labels)
        ) / 2

        return total_loss
