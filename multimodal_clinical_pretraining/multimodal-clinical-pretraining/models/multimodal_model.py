import torch

from torch import nn
import torch.nn.functional as F

from multimodal_clinical_pretraining.loss import CLIP_Loss


class MultiModalModel(nn.Module):
    def __init__(self, models, args):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.use_projector = args.use_projector
        self.measurement_mask_rate = args.measurement_mask_rate
        self.notes_mask_rate = args.notes_mask_rate
        self.use_cls_token = args.use_cls_token

        if args.use_projector:
            self.projector = Projector(args.mlp)
            self.loss = args.loss

            if args.loss == "CLIP":
                self.crit = CLIP_Loss(
                    world_size=args.world_size,
                    temperature=args.temp,
                    learnable_temp=args.learnable_temp,
                )
            elif args.loss == "CLIP+MSE":
                self.clip_crit = CLIP_Loss(
                    world_size=args.world_size,
                    temperature=args.temp,
                    learnable_temp=args.learnable_temp,
                )
                self.mse_crit1 = nn.SmoothL1Loss(reduction="none")
                self.mse_crit2 = nn.NLLLoss(reduction="none")

                self.final_layer1 = nn.Linear(
                    args.measurement_emb_size, args.n_features
                )
                self.final_layer2 = nn.Linear(args.notes_emb_size, args.vocab_size)

                self.softmax = nn.LogSoftmax(dim=-1)
                self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, xs):
        measurement_inf_mask = xs[0]["x"] != float("inf")

        xs[0]["x"][measurement_inf_mask] = xs[0]["x"][measurement_inf_mask].masked_fill(
            xs[0]["x"][measurement_inf_mask] > 5, 0
        )

        orig_xs = [x["x"] for x in xs]

        if self.use_projector:
            if self.training:
                measurement_mask = (xs[0]["x"] != float("inf")) & (
                    torch.rand_like(xs[0]["x"][:, 0, None]) < self.measurement_mask_rate
                )
                note_mask = (xs[1]["x"] != 0) & (
                    torch.rand_like(xs[1]["x"].float()) < self.notes_mask_rate
                )

                xs[0]["x"] = xs[0]["x"].masked_fill(measurement_mask, 0)
                xs[1]["x"] = xs[1]["x"].masked_fill(note_mask, 103)

                xs[0]["x"][measurement_inf_mask] = xs[0]["x"][
                    measurement_inf_mask
                ].masked_fill(xs[0]["x"][measurement_inf_mask] > 5, 0)

                assert torch.isfinite(xs[0]["x"]).any()

        zs = [model(**x) for model, x in zip(self.models, xs)]
        assert torch.isfinite(zs[0]).any()
        orig_zs = list(zs)

        if self.use_cls_token:
            zs = [zs[0][:, 0], zs[1][:, 0]]

        if self.use_projector:
            embeddings = [self.projector(z) for z in zs]

            if self.loss == "CLIP":
                embeddings = [F.normalize(emb, dim=-1) for emb in embeddings]
                return [
                    self.crit(embeddings[0], embeddings[1]),
                    self.crit.temperature.item(),
                    self.crit.temperature.item(),
                ]
            elif self.loss == "MSE":
                recon_loss1 = nn.functional.cross_entropy(
                    embeddings[0].transpose(1, 2), xs[0]["x"]
                )
                recon_loss2 = nn.functional.cross_entropy(
                    embeddings[1].transpose(1, 2), xs[1]["x"]
                )
                return (
                    recon_loss1 + recon_loss2,
                    0,
                    0,
                )
            elif self.loss == "CLIP+MSE":
                embeddings = [F.normalize(emb, dim=-1) for emb in embeddings]
                clip_loss = self.clip_crit(embeddings[0], embeddings[1])

                proj3 = self.final_layer1(orig_zs[0])
                proj4 = self.softmax(self.final_layer2(orig_zs[1]))
                orig_xs[0] = torch.cat(
                    [
                        self.models[0]
                        .cls_token.repeat(orig_xs[0].size(0), 1, 1)
                        .clone()
                        .detach(),
                        orig_xs[0],
                    ],
                    dim=1,
                )
                recon_loss1 = self.mse_crit1(proj3, orig_xs[0])
                recon_loss2 = self.mse_crit2(proj4.transpose(1, 2), orig_xs[1])

                measurement_mask = torch.cat(
                    [
                        torch.ones_like(measurement_mask)[:, 0, None],
                        measurement_mask,
                    ],
                    dim=1,
                )

                recon_loss1 = recon_loss1[measurement_mask].mean()
                recon_loss2 = recon_loss2[note_mask].mean()

                return (
                    clip_loss + recon_loss1 + recon_loss2,
                    self.clip_crit.temperature.item(),
                    self.clip_crit.temperature.item(),
                )
        else:
            return torch.cat(zs, dim=1)


def Projector(mlp="8192-8192-8192"):
    mlp_spec = mlp
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    if len(f) == 1:
        return Identity()

    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
