import torch
from torch import nn

from transformers import AutoModel, logging

logging.set_verbosity_error()


class ClinicalBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            z = self.model(x).last_hidden_state

        return z
