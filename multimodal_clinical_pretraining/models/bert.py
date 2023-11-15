import torch
from torch import nn

from transformers import BertModel, BertConfig
from transformers import logging

logging.set_verbosity_error()


class BERT(nn.Module):
    def __init__(self, output_size, args):
        super().__init__()

        config = BertConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.notes_emb_size,
            num_hidden_layers=args.notes_num_layers,
            num_attention_heads=args.notes_num_heads,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=args.notes_dropout,
            attention_probs_dropout_prob=args.notes_dropout,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=args.pad_token_id,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
        )

        self.model = BertModel(config)
        self.linear = nn.Parameter(torch.empty((args.notes_emb_size, output_size)))

        nn.init.kaiming_normal_(self.linear)

    def forward(self, x):
        z = self.model(x).last_hidden_state
        return z
