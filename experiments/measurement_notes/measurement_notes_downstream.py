import os
import sys
import warnings

import numpy as np

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

from torchmimic.utils import pad_colalte
from torchmimic.data.preprocessing import Normalizer
from torchmimic.data import (
    IHMDataset,
    PhenotypingDataset,
)
from torchmimic.loggers import (
    IHMLogger,
    PhenotypingLogger,
)

from downstream_argparser import parser

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, "../.."))

if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)


warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from multimodal_clinical_pretraining.data.utils import ShuffleTransform
from multimodal_clinical_pretraining.utils import load_pretrained_model
from multimodal_clinical_pretraining.scheduler import create_scheduler
from multimodal_clinical_pretraining.models import create_model


def train(args, train_dataloader, val_dataloader, test_dataloader):
    model = create_model(args)

    if args.pretrained_path is not None:
        model = load_pretrained_model(model, args)

    model = nn.Sequential(
        model,
        nn.Linear(args.measurement_emb_size, args.n_classes),
    ).cuda()
    model[-1].weight.data.normal_(mean=0.0, std=0.01)
    model[-1].bias.data.zero_()
    model = model.cuda()
    print(model)

    if args.experiment == "full_eval":
        params = model.parameters()
    elif args.experiment == "linear_eval":
        params = model[-1].parameters()
        model[0].eval()
    elif args.experiment in [
        "semi_0_5_eval",
        "semi_0_1_eval",
        "semi_0_01_eval",
        "full_eval",
    ]:
        if args.pretrained_path is not None:
            params = [
                {"params": model[0].parameters(), "lr": args.lr},
                {"params": model[-1].parameters(), "lr": args.linear_lr},
            ]
        else:
            params = model.parameters()

    optimizer = optim.AdamW(
        params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )

    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.task == "IHM":
        logger = IHMLogger(args.exp_name, args, log_wandb=True)
    elif args.task == "Phenotyping":
        logger = PhenotypingLogger(args.exp_name, args, log_wandb=True)

    criteria = nn.BCELoss()
    for epoch in range(args.epochs):
        ys = []
        preds = []
        print(f"LR = {optimizer.param_groups[0]['lr']}")
        if args.experiment in [
            "full_eval",
            "semi_0_5_eval",
            "semi_0_1_eval",
            "semi_0_01_eval",
        ]:
            model.train()
        else:
            model[-1].train()

        logger.reset()

        for values, labels, seq_lengths, _ in train_dataloader:
            optimizer.zero_grad()

            # Prepare measurement data
            measurement_x = values.cuda()
            labels = labels.cuda()

            input_list = []

            if args.use_measurements:
                input_list.append(
                    {
                        "x": measurement_x,
                    }
                )

            seq_lengths = torch.LongTensor(seq_lengths)
            logits = model(input_list)[:, 0, :].contiguous()
            logits = F.sigmoid(logits)

            if args.task == "IHM":
                logits = logits[:, 0]

            y = labels

            loss = criteria(logits, y)
            loss.backward()
            optimizer.step()

            pred = logits

            ys.append(y.detach().cpu())
            preds.append(pred.detach().cpu())

            logger.update(pred, y, loss)

        lr_scheduler.step(epoch + 1)
        logger.print_metrics(epoch, split="Train")
        logger.reset()

        preds = np.concatenate(preds)
        ys = np.concatenate(ys)

        model.eval()

        ys = []
        preds = []

        with torch.no_grad():
            for values, labels, seq_lengths, _ in val_dataloader:
                input_list = []

                # Prepare measurement data
                measurement_x = values.cuda()
                labels = labels.cuda()

                if args.use_measurements:
                    input_list.append(
                        {
                            "x": measurement_x,
                        }
                    )

                logits = model(input_list)[:, 0, :].contiguous()
                logits = F.sigmoid(logits)

                if args.task == "IHM":
                    logits = logits[:, 0]

                y = labels

                loss = criteria(logits, y)
                pred = logits

                ys.append(y.detach().cpu())
                preds.append(pred.detach().cpu())

                logger.update(pred, y, loss)

        logger.print_metrics(epoch, split="Eval")
        logger.reset()

        preds = np.concatenate(preds)
        ys = np.concatenate(ys)

    model.eval()

    ys = []
    preds = []

    with torch.no_grad():
        for values, labels, seq_lengths, _ in test_dataloader:
            input_list = []

            # Prepare measurement data
            measurement_x = values.cuda()
            labels = labels.cuda()

            if args.use_measurements:
                input_list.append(
                    {
                        "x": measurement_x,
                    }
                )

            logits = model(input_list)[:, 0, :].contiguous()
            logits = F.sigmoid(logits)

            if args.task == "IHM":
                logits = logits[:, 0]

            y = labels

            loss = criteria(logits, y)
            pred = logits

            ys.append(y.detach().cpu())
            preds.append(pred.detach().cpu())

            logger.update(pred, y, loss)

    logger.print_metrics(epoch, split="Test")

    preds = np.concatenate(preds)
    ys = np.concatenate(ys)


if __name__ == "__main__":
    activation_map = {"GELU": nn.GELU(), "ReLU": nn.ReLU()}
    args = parser()
    args.measurement_activation = activation_map[args.measurement_activation]

    args.use_projector = False

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True

    if args.experiment == "semi_0_1_eval":
        listfile = "10percent_"
    elif args.experiment == "semi_0_01_eval":
        listfile = "1percent_"
    elif args.experiment == "semi_0_5_eval":
        listfile = "50percent_"
    else:
        listfile = ""

    train_measurement_transform = transforms.Compose(
        [
            ShuffleTransform(args.measurement_max_seq_len),
        ]
    )

    test_measurement_transform = transforms.Compose(
        [
            ShuffleTransform(args.measurement_max_seq_len),
        ]
    )

    print(listfile)
    if args.task == "IHM":
        root = os.path.join(args.mimic_benchmark_root, "in-hospital-mortality")
        train_listfile = listfile + "train_listfile.csv"
        val_listfile = "val_listfile.csv"
        test_listfile = "test_listfile.csv"
        train_dataset = IHMDataset(
            root, customListFile=os.path.join(root, train_listfile), train=True
        )
        val_dataset = IHMDataset(
            root, customListFile=os.path.join(root, val_listfile), train=True
        )
        test_dataset = IHMDataset(
            root, customListFile=os.path.join(root, test_listfile), train=False
        )
    elif args.task == "Phenotyping":
        root = os.path.join(args.mimic_benchmark_root, "phenotyping")
        train_listfile = listfile + "train_listfile.csv"
        val_listfile = "val_listfile.csv"
        test_listfile = "test_listfile.csv"
        train_dataset = PhenotypingDataset(
            root,
            customListFile=os.path.join(root, train_listfile),
            train=True,
            transform=ShuffleTransform(args.measurement_max_seq_len),
        )
        val_dataset = PhenotypingDataset(
            root, customListFile=os.path.join(root, val_listfile), train=True
        )
        test_dataset = PhenotypingDataset(
            root, customListFile=os.path.join(root, test_listfile), train=False
        )

    discretizer_header = train_dataset.discretizer.transform(
        train_dataset.reader.read_example(0)["X"]
    )[1].split(",")
    cont_channels = [
        i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
    ]

    train_dataset.normalizer = Normalizer(fields=cont_channels)
    train_dataset.normalizer.load_params(
        "./multimodal_clinical_pretraining/resources/normalizer_params"
    )

    val_dataset.normalizer = Normalizer(fields=cont_channels)
    val_dataset.normalizer.load_params(
        "./multimodal_clinical_pretraining/resources/normalizer_params"
    )

    test_dataset.normalizer = Normalizer(fields=cont_channels)
    test_dataset.normalizer.load_params(
        "./multimodal_clinical_pretraining/resources/normalizer_params"
    )

    print(f"Length of training dataset = {len(train_dataset)}")
    print(f"Length of test dataset = {len(test_dataset)}")

    args.n = len(train_dataset)
    args.n_features = 76

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=pad_colalte,
        pin_memory=False,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        num_workers=0,
        collate_fn=pad_colalte,
        pin_memory=False,
        shuffle=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        num_workers=0,
        collate_fn=pad_colalte,
        pin_memory=False,
        shuffle=False,
    )

    args.use_measurements = True
    args.use_notes = False

    if args.task == "IHM":
        args.n_classes = 1

    elif args.task == "Phenotyping":
        args.n_classes = 25

    torch.manual_seed(42)
    np.random.seed(42)

    output = train(
        args,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )
