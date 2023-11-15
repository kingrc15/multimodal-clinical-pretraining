from torch import nn

from .transformer import MultiheadAttention
from .bert import BERT
from .clinicalbert import ClinicalBERT
from .multimodal_model import MultiModalModel


def create_model(args):
    model_list = []

    if args.use_measurements:
        model_list.append(get_measurement_model(args))

    if args.use_notes:
        model_list.append(get_notes_model(args))

    model = MultiModalModel(
        model_list,
        args,
    ).cuda()

    return model


def get_notes_model(args):
    if args.text_model == "BERT":
        notes_model = BERT(output_size=int(args.mlp.split("-")[-1]), args=args)
    elif args.text_model == "ClinicalBERT":
        notes_model = ClinicalBERT()
        notes_model.embed_dim = 768
    else:
        raise NotImplementedError

    return notes_model


def get_measurement_model(args):
    if args.measurement_model == "Transformer":
        measurement_model = MultiheadAttention(
            n_features=args.n_features,
            embed_dim=args.measurement_emb_size,
            num_heads=args.measurement_num_heads,
            dropout=args.measurement_dropout,
            num_layers=args.measurement_num_layers,
            use_pos_emb=args.use_pos_emb,
            activation=args.measurement_activation,
        )
    else:
        raise NotImplementedError

    return measurement_model
