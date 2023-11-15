import os
import sys

import time
import warnings
import datetime
import numpy as np
import wandb


from sklearn.metrics import roc_auc_score, average_precision_score

import torch
from torch import nn
import torch.multiprocessing
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchvision import transforms

from torchmimic.data import IHMDataset
from torchmimic.utils import pad_colalte

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, "../.."))

if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

from pretrain_argparser import parser

from multimodal_clinical_pretraining.optim import create_optimizer
from multimodal_clinical_pretraining.scheduler import create_scheduler
from multimodal_clinical_pretraining.models import create_model
from multimodal_clinical_pretraining.distributed_utils import init_distributed_mode

from multimodal_clinical_pretraining.data.utils import (
    ShuffleTransform,
    multimodal_pad_collate_fn,
)
from multimodal_clinical_pretraining.data import (
    MIMICIIINoteDataset,
    MIMICIIIBenchmarkDataset,
)


warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(
    args,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    val_zeroshot_dataloader,
    test_zeroshot_dataloader,
):
    print(args)
    gpu = torch.device(args.device)

    exp_path = os.path.join("exp_outputs", args.exp_name)

    if args.rank == 0:
        if not os.path.exists("exp_outputs"):
            os.mkdir("exp_outputs")
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)

        wandb.init(
            project="Multimodal Clinical Pretraining",
            name=args.exp_name,
            config=args,
            save_code=True,
            settings=wandb.Settings(code_dir=exp_path),
        )

    model = create_model(args)

    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    print(model)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True
    )
    model_without_ddp = model.module

    best_auc = -float("inf")
    scaler = torch.cuda.amp.GradScaler()
    cont_loss = 0

    for epoch in range(args.epochs):
        print(f"LR = {optimizer.param_groups[0]['lr']}")
        model.train()
        train_dataloader.sampler.set_epoch(epoch)

        for idx, notes_dict in enumerate(train_dataloader):
            print(f"[{idx+1}/{len(train_dataloader)}]", end="\r")

            optimizer.zero_grad()

            measurement_input = {}

            measurement_input["x"] = notes_dict["measurement_data"].cuda(
                gpu, non_blocking=True
            )

            notes_input = {
                "x": notes_dict["note_texts_tokenized"].cuda(gpu, non_blocking=True)
            }

            model_input = [measurement_input, notes_input]

            with torch.cuda.amp.autocast(), torch.backends.cuda.sdp_kernel(
                enable_flash=True
            ):
                loss = model(model_input)

            avg_measurements_tau = loss[1]
            avg_notes_tau = loss[2]
            loss = loss[0]

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            torch.cuda.empty_cache()

            if args.rank == 0:
                cont_loss = loss.item() * 0.1 + cont_loss * 0.9

                if idx % args.report_freq == 0 and idx != 0:
                    result_str = f"Train Epoch {epoch + 1}: "
                    result_str += f"Loss = {cont_loss:.6f}, "
                    result_str += (
                        f"Average Measurement Tau = {avg_measurements_tau:.6f}, "
                    )
                    result_str += f"Average Notes Tau = {avg_notes_tau:.6f}"
                    print(result_str)

        lr_scheduler.step(epoch + 1)

        if (epoch + 1) % 5 == 0:
            score_test_m2t, score_test_t2m, eval_loss = evaluation(
                model_without_ddp,
                val_dataloader,
                gpu,
                args,
            )

            if args.rank == 0:
                result_str = f"Train Epoch {epoch+1}: "
                result_str += f"Loss = {cont_loss:.6f}, "
                result_str += f"Average Measurement Tau = {avg_measurements_tau:.6f}, "
                result_str += f"Average Notes Tau = {avg_notes_tau:.6f}"
                print(result_str)

                eval_result = {}
                eval_result = itm_eval(score_test_m2t, score_test_t2m)

                (
                    zeroshot_auc_roc,
                    zeroshot_auc_pr,
                    zeroshot_bal_acc,
                ) = zeroshot_eval(
                    model_without_ddp,
                    val_zeroshot_dataloader,
                    train_dataloader.dataset.tokenizer,
                )

                eval_result["Eval Zeroshot Balanced Accuracy"] = zeroshot_bal_acc
                eval_result["Eval Zeroshot AUC-ROC"] = zeroshot_auc_roc
                eval_result["Eval Zeroshot AUC-PR"] = zeroshot_auc_pr

                result_str = f"Eval Epoch {epoch + 1}: "

                for key, item in eval_result.items():
                    result_str += f"{key}: {item}, "

                result_str = result_str[:-2]
                print(result_str)

                if zeroshot_auc_roc > best_auc:
                    best_auc = zeroshot_auc_roc
                    torch.save(
                        model.state_dict(),
                        os.path.join(exp_path, "best_model.pth"),
                    )

                torch.save(
                    model.state_dict(),
                    os.path.join(exp_path, f"model_{epoch + 1}.pth"),
                )
                eval_result["Epoch"] = epoch + 1
                eval_result["Train Loss"] = cont_loss
                eval_result["Eval Loss"] = eval_loss
                eval_result["Average Measurement Tau"] = avg_measurements_tau
                eval_result["Average Notes Tau"] = avg_notes_tau

                wandb.log(eval_result)

    model.load_state_dict(torch.load(os.path.join(exp_path, "best_model.pth")))

    score_test_m2t, score_test_t2m, eval_loss = evaluation(
        model_without_ddp,
        test_dataloader,
        gpu,
        args,
    )

    if args.rank == 0:
        result_str = f"Train Epoch {epoch+1}: "
        result_str += f"Loss = {cont_loss:.6f}, "
        result_str += f"Average Measurement Tau = {avg_measurements_tau:.6f}, "
        result_str += f"Average Notes Tau = {avg_notes_tau:.6f}"
        print(result_str)

        eval_result = {}
        eval_result = itm_eval(score_test_m2t, score_test_t2m)

        (
            zeroshot_auc_roc,
            zeroshot_auc_pr,
            zeroshot_bal_acc,
        ) = zeroshot_eval(
            model_without_ddp,
            test_zeroshot_dataloader,
            train_dataloader.dataset.tokenizer,
        )

        eval_result["Test Zeroshot Balanced Accuracy"] = zeroshot_bal_acc
        eval_result["Test Zeroshot AUC-ROC"] = zeroshot_auc_roc
        eval_result["Test Zeroshot AUC-PR"] = zeroshot_auc_pr

        result_str = "Test: "

        for key, item in eval_result.items():
            result_str += f"{key}: {item}, "

        result_str = result_str[:-2]
        print(result_str)

        torch.save(
            model.state_dict(),
            os.path.join(exp_path, f"model_{epoch + 1}.pth"),
        )
        eval_result["Epoch"] = epoch + 1
        eval_result["Test Loss"] = eval_loss
        eval_result["Average Measurement Tau"] = avg_measurements_tau
        eval_result["Average Notes Tau"] = avg_notes_tau

        wandb.log(eval_result)

    if args.rank == 0:
        wandb.finish()


@torch.no_grad()
def zeroshot_eval(
    model,
    zeroshot_dataloader,
    tokenizer,
):
    model.eval()

    print("Zeroshot evaluation...")
    start_time = time.time()

    positive_txt = ["patient deceased"]
    negative_txt = ["discharged today"]

    tokenized_text = (
        pad_sequence(
            [
                torch.LongTensor(tokenizer.encode(txt)).t()
                for txt in positive_txt + negative_txt
            ]
        )
        .t()
        .cuda()
    )

    note_representations = torch.cat(
        [model.models[1](tokenized_text)],
    )

    accuracy_pos = 0
    count_pos = 0
    accuracy_neg = 0
    count_neg = 0

    bin_pred = []
    bin_label = []

    for idx, (values, labels, _, _) in enumerate(zeroshot_dataloader):
        print(f"[{idx+1}/{len(zeroshot_dataloader)}]", end="\r")

        measurement_x = {"x": values.cuda()}

        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            measurement_representations = model.models[0](**measurement_x)

            if model.use_cls_token:
                zs = [measurement_representations[:, 0], note_representations[:, 0]]
            else:
                zs = [measurement_representations, note_representations]

            embeddings = [F.normalize(emb, dim=-1) for emb in zs]
            sim = (
                torch.einsum("i d, j d -> i j", embeddings[0], embeddings[1])
                / model.clip_crit.temperature
            )

            pred = (sim.argmax(dim=1) < len(positive_txt)).cpu()

            if labels.item() == 1:
                accuracy_pos += int(labels == pred)
                count_pos += 1
            else:
                accuracy_neg += int(labels == pred)
                count_neg += 1

            embeddings = [F.normalize(emb, dim=-1) for emb in zs]
            sim = (
                torch.einsum("i d, j d -> i j", embeddings[0], embeddings[1])
                / model.clip_crit.temperature
            )

            sim = sim[0]
            sim = torch.nn.functional.softmax(sim)

            bin_pred.append(sim[0].cpu())
            bin_label.append(labels.item())

    auc_roc = roc_auc_score(bin_label, bin_pred)
    auc_pr = average_precision_score(bin_label, bin_pred)
    bal_acc = (accuracy_pos / count_pos + accuracy_neg / count_neg) / 2

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return auc_roc, auc_pr, bal_acc


@torch.no_grad()
def evaluation(model, data_loader, device, args):
    model.eval()

    print("Computing features for evaluation...")
    start_time = time.time()

    measurement_embeddings = []
    text_embeddings = []

    for idx, notes_dict in enumerate(data_loader):
        print(f"[{idx+1}/{len(data_loader)}]", end="\r")
        if len(measurement_embeddings) * args.batch_size > 4096:
            break

        measurement_input = {}
        measurement_input["x"] = notes_dict["measurement_data"].cuda(
            device, non_blocking=True
        )

        notes_input = {
            "x": notes_dict["note_texts_tokenized"].cuda(device, non_blocking=True),
        }

        model_input = [measurement_input, notes_input]

        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            zs = [model(**x) for model, x in zip(model.models, model_input)]

            if model.use_cls_token:
                zs = [z[:, 0] for z in zs]

            embeddings = [F.normalize(emb, dim=-1) for emb in zs]
            sim = torch.einsum("i d, j d -> i j", embeddings[1], embeddings[0])
            labels = torch.arange(embeddings[0].shape[0], device=embeddings[0].device)
            loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        measurement_embeddings.append(zs[0])
        text_embeddings.append(zs[1])

    measurement_embeddings = torch.concat(measurement_embeddings, dim=0)
    text_embeddings = torch.concat(text_embeddings, dim=0)

    measurement_embeddings = F.normalize(measurement_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    score_matrix_m2t = torch.full(
        (
            len(measurement_embeddings),
            len(measurement_embeddings),
        ),
        -100.0,
    ).to(device)

    sims_matrix = (
        measurement_embeddings @ text_embeddings.T / model.clip_crit.temperature
    )

    num_tasks = dist.get_world_size()
    rank = dist.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
        score_matrix_m2t[start + i, topk_idx] = topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2m = torch.full(
        (
            len(measurement_embeddings),
            len(measurement_embeddings),
        ),
        -100.0,
    ).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=args.k_test, dim=0)
        score_matrix_t2m[start + i, topk_idx] = topk_sim

    dist.barrier()
    torch.distributed.all_reduce(score_matrix_m2t, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(score_matrix_t2m, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return score_matrix_m2t.cpu().numpy(), score_matrix_t2m.cpu().numpy(), loss


@torch.no_grad()
def itm_eval(scores_m2t, scores_t2m):
    # Measurement->Text
    ranks = np.zeros(scores_m2t.shape[0])
    for index, score in enumerate(scores_m2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    ranks = np.zeros(scores_t2m.shape[0])

    for index, score in enumerate(scores_t2m):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    mr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    mr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    mr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    mr_mean = (mr1 + mr5 + mr10) / 3
    r_mean = (tr_mean + mr_mean) / 2

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "meas_r1": mr1,
        "meas_r5": mr5,
        "meas_r10": mr10,
        "meas_r_mean": mr_mean,
        "r_mean": r_mean,
    }
    return eval_result


if __name__ == "__main__":
    torch.cuda.empty_cache()
    activation_map = {"GELU": nn.GELU(), "ReLU": nn.ReLU(inplace=True)}

    args = parser()
    args.measurement_activation = activation_map[args.measurement_activation]
    args.use_projector = True

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True

    init_distributed_mode(args)

    seed = 42
    torch.manual_seed(seed + args.rank)
    np.random.seed(seed + args.rank)

    # You can you any of these by uncommenting the notes you want
    used_note_types = [
        # "Echo",
        # "ECG",
        "Nursing",
        "Physician ",
        # "Rehab Services",
        # "Case Management ",
        "Respiratory ",
        # "Nutrition",
        # "General",
        # "Social Work",
        # "Pharmacy",
        # "Consult",
        "Radiology",
        "Nursing/other",
        "Discharge summary",
    ]

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

    train_measurement_dataset = MIMICIIIBenchmarkDataset(
        mimic3_benchmark_root=os.path.join(args.mimic_benchmark_root, "root"),
        mimic3_root=args.mimic_root,
        split="train",
        transform=train_measurement_transform,
    )

    val_measurement_dataset = MIMICIIIBenchmarkDataset(
        mimic3_benchmark_root=os.path.join(args.mimic_benchmark_root, "root"),
        mimic3_root=args.mimic_root,
        split="valid",
        transform=test_measurement_transform,
    )

    test_measurement_dataset = MIMICIIIBenchmarkDataset(
        mimic3_benchmark_root=os.path.join(args.mimic_benchmark_root, "root"),
        mimic3_root=args.mimic_root,
        split="test",
        transform=test_measurement_transform,
    )

    train_notes_dataset = MIMICIIINoteDataset(
        root=args.mimic_root,
        split="train",
        max_seq_len=args.notes_max_seq_len,
        mask_rate=0,
        collate_fn=multimodal_pad_collate_fn,
        max_instances=10000,
        used_note_types=used_note_types,
        measurement_dataset=train_measurement_dataset,
    )

    val_notes_dataset = MIMICIIINoteDataset(
        root=args.mimic_root,
        split="valid",
        max_seq_len=args.notes_max_seq_len,
        mask_rate=0,
        collate_fn=multimodal_pad_collate_fn,
        max_instances=10000,
        used_note_types=used_note_types,
        measurement_dataset=val_measurement_dataset,
    )

    test_notes_dataset = MIMICIIINoteDataset(
        root=args.mimic_root,
        split="test",
        max_seq_len=args.notes_max_seq_len,
        mask_rate=0,
        collate_fn=multimodal_pad_collate_fn,
        max_instances=10000,
        used_note_types=used_note_types,
        measurement_dataset=test_measurement_dataset,
    )

    print(f"Length of training dataset = {len(train_notes_dataset)}")
    print(f"Length of val dataset = {len(val_notes_dataset)}")
    print(f"Length of test dataset = {len(test_notes_dataset)}")

    root = args.mimic_benchmark_root + "in-hospital-mortality"
    val_listfile = "val_listfile.csv"
    test_listfile = "test_listfile.csv"

    val_zeroshot_dataset = IHMDataset(
        root, customListFile=os.path.join(root, val_listfile), train=True
    )
    val_zeroshot_dataset = IHMDataset(
        root, customListFile=os.path.join(root, test_listfile), train=False
    )

    val_zeroshot_dataloader = DataLoader(
        val_zeroshot_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=pad_colalte,
        pin_memory=True,
        shuffle=False,
    )

    test_zeroshot_dataloader = DataLoader(
        val_zeroshot_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=pad_colalte,
        pin_memory=True,
        shuffle=False,
    )

    args.n = len(train_notes_dataset)
    args.vocab_size = train_notes_dataset.tokenizer.vocab_size
    args.n_features = train_measurement_dataset.n_features

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_notes_dataset, shuffle=True
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_notes_dataset, shuffle=False
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_notes_dataset, shuffle=False
    )

    train_dataloader = DataLoader(
        train_notes_dataset,
        batch_size=per_device_batch_size,
        num_workers=8,
        collate_fn=train_notes_dataset.collate,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    val_dataloader = DataLoader(
        val_notes_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=val_notes_dataset.collate,
        pin_memory=True,
        sampler=val_sampler,
    )

    test_dataloader = DataLoader(
        test_notes_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=test_notes_dataset.collate,
        pin_memory=True,
        sampler=test_sampler,
    )

    tokenizer = train_notes_dataset.tokenizer

    args.sep_token_id = tokenizer.sep_token_id
    args.pad_token_id = tokenizer.pad_token_id
    args.bos_token_id = tokenizer.bos_token_id
    args.eos_token_id = tokenizer.eos_token_id
    args.cls_token_id = tokenizer.cls_token_id

    main(
        args,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        val_zeroshot_dataloader,
        test_zeroshot_dataloader,
    )
