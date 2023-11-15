import argparse


def parser():
    parser = argparse.ArgumentParser("multimodal_downstream")

    # Experiment
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument(
        "--experiment",
        choices=[
            "semi_0_5_eval",
            "semi_0_1_eval",
            "semi_0_01_eval",
            "full_eval",
            "linear_eval",
        ],
        default="linear_eval",
    )
    parser.add_argument("--task", choices=["IHM", "Phenotyping"], default="IHM")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument(
        "--report_freq", type=float, default=100, help="report frequency"
    )

    # Data
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--mimic_root", type=str, default="/ssd-data/datasets/MIMIC-III/1.4/"
    )
    parser.add_argument(
        "--mimic_benchmark_root",
        type=str,
        default="/ssd-data/datasets/mimic3-benchmarks/",
    )

    # Training
    parser.add_argument("--opt", default="adamW")
    parser.add_argument("--sched", default="cosine")
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--linear_lr", default=0.001, type=float)
    parser.add_argument("--min_lr", default=1e-8, type=float)
    parser.add_argument("--warmup", default=True, type=bool)
    parser.add_argument("--warmup_lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=0.02, type=float)
    parser.add_argument("--decay_rate", default=1, type=float)
    parser.add_argument("--warmup_epochs", default=20, type=int)
    parser.add_argument("--cooldown_epochs", default=0, type=int)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=5)

    # Multimodal Model
    parser.add_argument("--use_cls_token", action="store_true")
    parser.add_argument("--mlp", default="128")

    # Measurement Model
    parser.add_argument("--use_measurements", action="store_true")
    parser.add_argument(
        "--measurement_model", default="Transformer", choices=["Transformer"]
    )
    parser.add_argument("--measurement_emb_size", type=int, default=64)
    parser.add_argument("--measurement_num_heads", type=int, default=16)
    parser.add_argument("--measurement_num_layers", type=int, default=8)
    parser.add_argument("--use_pos_emb", action="store_true")
    parser.add_argument(
        "--measurement_activation", choices=["ReLU", "GELU"], default="GELU"
    )
    parser.add_argument("--measurement_mask_rate", type=float, default=0.1)
    parser.add_argument("--measurement_max_seq_len", type=int, default=256)
    parser.add_argument("--measurement_dropout", type=float, default=0.1)

    # Notes Model
    parser.add_argument("--text_model", default="BERT", choices=["BERT"])
    parser.add_argument("--use_notes", action="store_true")
    parser.add_argument("--notes_emb_size", type=int, default=64)
    parser.add_argument("--notes_num_heads", type=int, default=16)
    parser.add_argument("--notes_num_layers", type=int, default=8)
    parser.add_argument("--notes_mask_rate", type=float, default=0.4)
    parser.add_argument("--notes_dropout", type=float, default=0.1)
    parser.add_argument("--notes_max_seq_len", type=int, default=256)

    parser.add_argument("--device", default="cuda")

    return parser.parse_args()
