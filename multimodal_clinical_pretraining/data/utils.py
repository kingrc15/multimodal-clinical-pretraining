import os

import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence


class CustomBins:
    inf = 1e18
    bins = [
        (-inf, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 14),
        (14, +inf),
    ]
    nbins = len(bins)
    means = [
        11.450379,
        35.070846,
        59.206531,
        83.382723,
        107.487817,
        131.579534,
        155.643957,
        179.660558,
        254.306624,
        585.325890,
    ]


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = torch.zeros((CustomBins.nbins,))
                ret[i] = 1
                return int(ret)
            return i
    return None


class MaskTransform:
    def __init__(self, p=0.0, value=0):
        self.p = p
        self.value = value

    def __call__(self, x):
        mask = torch.rand_like(x) > self.p
        x.masked_fill_(mask, self.value)
        return x


class UseLastTransform:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def __call__(self, x):
        return x[-self.max_seq_len :]


class ConstValueImpute:
    def __init__(self, values):
        super().__init__()
        self.values = values

    def __call__(self):
        return self.values


class ShuffleTransform:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def __call__(self, x):
        window = len(x) - self.max_seq_len - 1
        if window > 0:
            start_idx = np.random.randint(0, window)
        else:
            start_idx = 0

        return x[start_idx : start_idx + self.max_seq_len]


class PartitionWindow:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def __call__(self, xs):
        x_list = []
        start_index = 0
        for x in xs:
            while start_idx < len(x):
                x_list.append(x[start_idx : self.max_seq_len])
                start_idx += self.max_seq_len

        return x_list


def triplet_collate(batch):
    (stay_ids, data, demo, sample_idx, ihm) = zip(*batch)

    demo = torch.stack(demo)
    data = pad_sequence(data, batch_first=True, padding_value=-float("inf"))
    sample_idx = torch.LongTensor(sample_idx)

    return {
        "stay_ids": stay_ids,
        "data": data,
        "demographics": demo,
        "sample_idx": sample_idx,
        "ihm": ihm,
    }


def multimodal_pad_collate_fn(batch, max_seq_len, pad_token_id):
    (
        note_text,
        note_text_tokenized,
        measurement_data,
        idx,
        measurement_idx,
    ) = zip(*batch)

    dt_list = []

    note_texts_tokenized = pad_sequence(
        note_text_tokenized, batch_first=True, padding_value=pad_token_id
    )

    measurement_data = pad_sequence(
        measurement_data, batch_first=True, padding_value=float("inf")
    )

    sample_idx = torch.LongTensor(idx)
    measurement_idx = torch.LongTensor(measurement_idx)

    output_dic = {
        "note_texts": note_text,
        "note_texts_tokenized": note_texts_tokenized,
        "measurement_data": measurement_data,
        "sample_idx": sample_idx,
        "measurement_idx": measurement_idx,
    }

    return output_dic


def mi_collate_fn(batch, max_seq_len, pad_token_id):
    (
        stay_ids,
        note_datetime,
        note_texts,
        note_texts_tokenized,
        seq_len,
        sample_idx,
        representations,
    ) = zip(*batch)

    indices = []
    tmp_token_list = []
    dt_list = []

    for i in range(len(note_texts_tokenized)):
        for j in range(len(note_texts_tokenized[i])):
            indices.append(i)
            tmp_token_list.append(note_texts_tokenized[i][j])
            dt_list.append(note_datetime[i][j])

    note_texts_tokenized = pad_sequence(
        tmp_token_list, batch_first=True, padding_value=pad_token_id
    )

    sample_idx = torch.LongTensor(sample_idx)
    indices = torch.LongTensor(indices)

    if representations[0] is not None:
        representations = torch.cat(representations, dim=0)

    output_dic = {
        "stay_ids": stay_ids,
        "datetime": dt_list,
        "note_texts": note_texts,
        "note_texts_tokenized": note_texts_tokenized,
        "indices": indices,
        "seq_len": torch.LongTensor(seq_len),
        "sample_idx": sample_idx,
        "representations": representations,
    }

    return output_dic
