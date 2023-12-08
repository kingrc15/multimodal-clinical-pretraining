import os, glob, datetime, random
import numpy as np
import torch
import pandas as pd
import time
import pickle
import multiprocessing
from functools import partial

from torch.nn.utils.rnn import pad_sequence

from .measurement_preprocessing.preprocessing import Normalizer, Discretizer


class MIMICIIIBenchmarkDataset:
    def __init__(self, mimic3_benchmark_root, mimic3_root, split, transform=None):
        self.modality = "measurement"
        self.mimic3_benchmark_root = mimic3_benchmark_root
        self.mimic3_root = mimic3_root
        self.split = split
        self.n_features = 76
        self.transform = transform

        self.stay_ids = []
        self.data_df = pd.DataFrame(
            columns=["stay_ids", "intimes", "outtimes", "hadm_id", "paths"]
        )

        icu_stay_df = pd.read_csv(os.path.join(mimic3_root, "ICUSTAYS.csv"))

        split_file = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "resources",
                "valset.csv",
            ),
            header=None,
        )
        if split == "train":
            split_file = split_file[split_file[1] == 0]
        elif split == "valid":
            split_file = split_file[split_file[1] == 1]

        if split in ["train", "valid"]:
            stay_dirs = os.path.join(mimic3_benchmark_root, "train")
        elif split == "test":
            stay_dirs = os.path.join(mimic3_benchmark_root, "test")

        for directory in glob.glob(os.path.join(stay_dirs, "*")):
            files = glob.glob(os.path.join(directory, "*"))

            if (
                int(directory.split("/")[-1]) in split_file[0]
                and split
                in [
                    "train",
                    "valid",
                ]
            ) or split == "test":
                for file in files:
                    if "timeseries" in file:
                        try:
                            data = self._read_timeseries(file)[0]
                        except Exception as e:
                            continue

                        separated_filename = file.split("/")[-1].split("_")
                        if len(separated_filename) < 8:
                            continue
                        stay_id = int(separated_filename[0].replace("episode", ""))

                        separated_filename[-1] = separated_filename[-1].replace(
                            ".csv", ""
                        )

                        intime = self.string_to_datetime(
                            separated_filename[3], separated_filename[4]
                        )

                        outtime = self.string_to_datetime(
                            separated_filename[-2], separated_filename[-1]
                        )

                        stay_data = icu_stay_df[icu_stay_df["ICUSTAY_ID"] == stay_id]
                        hadm_id = int(stay_data["HADM_ID"].item())
                        subject_id = stay_data["SUBJECT_ID"].item()

                        new_row = pd.DataFrame(
                            data={
                                "stay_ids": stay_id,
                                "intimes": intime,
                                "outtimes": outtime,
                                "hadm_id": hadm_id,
                                "paths": file,
                            },
                            index=[0],
                        )

                        self.stay_ids.append(stay_id)
                        self.data_df = pd.concat(
                            [self.data_df, new_row], ignore_index=True
                        )

        self.discretizer = Discretizer(
            timestep=1.0,
            store_masks=True,
            impute_strategy="previous",
            start_time="zero",
            config_path=os.path.join(
                os.path.dirname(__file__),
                "..",
                "resources",
                "discretizer_config.json",
            ),
        )

        data = self._read_timeseries(self.data_df.iloc[0]["paths"])

        discretizer_header = self.discretizer.transform(data[0])[1].split(",")
        cont_channels = [
            i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
        ]

        self.normalizer = Normalizer(fields=cont_channels)

        self.normalizer.load_params(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "resources",
                "normalizer_params",
            )
        )

        self.data_df.set_index("stay_ids", inplace=True)

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(ts_filename, "r") as tsfile:
            header = tsfile.readline().strip().split(",")
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(",")
                data = np.array(mas)
                if float(data[0]) > 0:
                    ret.append(data)
        return (np.stack(ret), header)

    def string_to_datetime(self, date_str, time_str):
        date = date_str.split("-")
        year = int(date[0])
        month = int(date[1])
        day = int(date[2])

        time = time_str.split("-")
        hour = int(time[0])
        minute = int(time[1])
        second = int(time[2])

        return datetime.datetime(year, month, day, hour, minute, second)

    def get_by_stayid(self, stay_id):
        intime, outtime, hadm_id, path = self.data_df.loc[stay_id]

        with torch.no_grad():
            try:
                data = self._read_timeseries(path)[0]
            except Exception as e:
                print(path)
                raise e

            data, header = self.discretizer.transform(data)
            data = torch.Tensor(self.normalizer.transform(data))

            if self.transform is not None:
                data = self.transform(data)

            if not torch.isfinite(data).any():
                print(data, stay_id)
                raise

        return data, self.stay_ids.index(stay_id)

    def __getitem__(self, idx):
        stay_id = self.stay_ids[idx]
        intime, outtime, hadm_id, path = self.data_df.loc[stay_id]

        with torch.no_grad():
            try:
                data = self._read_timeseries(path)[0]
            except Exception as e:
                print(path)
                raise e

            data, header = self.discretizer.transform(data)
            data = torch.Tensor(self.normalizer.transform(data))
            ts = [
                intime + datetime.timedelta(hours=self.discretizer._timestep * x)
                for x in range(len(data[0]))
            ]

            data = torch.cat((torch.Tensor([list(range(data.size(0)))]).T, data), dim=1)
            if self.transform is not None:
                data = self.transform(data)
            dt = pd.Series(
                [t for i, t in enumerate(ts) if i in data[:, 0]]
            ).values.astype("datetime64[s]")
            data = data[:, 1:]

        return (
            stay_id,
            dt,
            data,
            data.shape[0],
            idx,
        )

    def __len__(self):
        return len(self.stay_ids)

    def remove_stays(self, stay_ids):
        stay_list = []
        for stay_idx, stay_id in enumerate(self.stay_ids):
            if stay_id not in stay_ids:
                stay_list.append(self.stay_ids[stay_idx])

        self.stay_ids = stay_list

    def collate(self, batch):
        (
            stay_ids,
            measurement_datetimes,
            values,
            seq_len,
            sample_idx,
        ) = zip(*batch)

        values = pad_sequence(
            values,
            batch_first=True,
            padding_value=float("inf"),
        ).float()

        if torch.any(torch.isnan(values)):
            assert 0, "values has nan."

        sample_idx = torch.LongTensor(sample_idx)

        output_dic = {
            "stay_ids": stay_ids,
            "measurement_datetime": measurement_datetimes,
            "values": values,
            "seq_len": torch.LongTensor(seq_len),
            "sample_idx": sample_idx,
        }

        return output_dic
