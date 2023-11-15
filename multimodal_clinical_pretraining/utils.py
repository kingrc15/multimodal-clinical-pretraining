import os
import torch


def load_pretrained_model(model, args):
    state_dict = torch.load(args.pretrained_path, map_location="cpu")

    backbone_state_dict = {}

    if args.use_notes and args.use_measurements:
        for key in state_dict.keys():
            if "models." in key:
                backbone_state_dict[key] = state_dict[key]

        backbone_state_dict = {
            key.replace("module.models.", ""): value
            for (key, value) in backbone_state_dict.items()
        }

        linear_state_dict = {}

        for key in state_dict.keys():
            if "linears." in key:
                linear_state_dict[key] = state_dict[key]

        linear_state_dict = {
            key.replace("module.linears.", ""): value
            for (key, value) in linear_state_dict.items()
        }

    elif args.use_measurements:
        for key in state_dict.keys():
            if "models.0" in key:
                backbone_state_dict[key] = state_dict[key]

        backbone_state_dict = {
            key.replace("module.models.", ""): value
            for (key, value) in backbone_state_dict.items()
        }

        linear_state_dict = {}

        for key in state_dict.keys():
            if "linears.0" in key:
                linear_state_dict[key] = state_dict[key]

        linear_state_dict = {
            key.replace("module.linears.", ""): value
            for (key, value) in linear_state_dict.items()
        }

    else:
        for key in state_dict.keys():
            if "models.1" in key:
                backbone_state_dict[key] = state_dict[key]

        backbone_state_dict = {
            key.replace("models.1", "models.0"): value
            for (key, value) in backbone_state_dict.items()
        }

        backbone_state_dict = {
            key.replace("module.models.", ""): value
            for (key, value) in backbone_state_dict.items()
        }

        linear_state_dict = {}

        for key in state_dict.keys():
            if "linears.1" in key:
                linear_state_dict[key] = state_dict[key]

        linear_state_dict = {
            key.replace("linears.1", "linears.0"): value
            for (key, value) in linear_state_dict.items()
        }

        linear_state_dict = {
            key.replace("module.linears.", ""): value
            for (key, value) in linear_state_dict.items()
        }

    model.models.load_state_dict(backbone_state_dict, strict=False)
    # model.linears.load_state_dict(linear_state_dict)

    return model
