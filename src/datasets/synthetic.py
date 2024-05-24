import os

import numpy as np
import torch


class SyntheticDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        nr_image_features,
        split="train",
        missing_mode="debug",
        infinite_flag=False,
        model_random_state=0,
        dataset_size=None,
        missing_factor=None,
        transforms=None,
    ):

        self.split = split
        self.infinite_flag = infinite_flag
        self.missing_mode = missing_mode
        self.transforms = transforms

        self.missing_factor = missing_factor
        self.dataset_size = dataset_size

        # set these parameters as in TAVI dataset
        if dataset_size is None:
            if split == "train":
                self.len_dataset = 1159
            elif split in ["val", "test"]:
                self.len_dataset = 145
            elif split == "trainval":
                self.len_dataset = 1304
            else:
                raise ValueError
        else:
            if split == "train":
                self.len_dataset = int(0.9 * dataset_size)
            elif split == "val":
                self.len_dataset = int(0.1 * dataset_size)
            elif split == "trainval":
                self.len_dataset = dataset_size
            elif split == "test":
                self.len_dataset = 1000
            else:
                raise ValueError
        self.nr_image_features = nr_image_features
        self.nr_cont_features = 10
        self.nr_cat_features = 15
        self.nr_aux = 15
        self.nr_table_features = self.nr_cont_features + self.nr_cat_features
        # get the Bernoulli p's for categorical features, close to actual values of TAVI dataset
        self.cat_table_prior = np.array(
            [
                0.18,
                0.15,
                0.21,
                0.73,
                0.25,
                0.56,
                0.29,
                0.52,
                0.81,
                0.93,
                0.21,
                0.20,
                0.03,
                0.31,
                0.04,
            ]
        )
        self.mean = np.zeros(self.nr_cont_features)
        self.std = np.ones(self.nr_cont_features)
        self.aux_mean = np.zeros(self.nr_aux)
        self.aux_std = np.ones(self.nr_aux)

        # let's define the model
        rs = np.random.RandomState(model_random_state)
        self.alpha_A = 0.1 * rs.randn(self.nr_table_features) + 0.1
        self.alpha_I = 0.1 * rs.randn(self.nr_image_features) - 0.1
        self.beta = 0.1 * rs.randn(self.nr_table_features, self.nr_image_features) + 0.1
        self.phi = 0.1 * rs.randn(self.nr_image_features, self.nr_aux) - 0.1
        self.b_Y = rs.uniform(-0.1, 0.1)
        self.log_var_I = rs.uniform(-0.1, 0.1)  # scalar
        self.log_var_J = rs.uniform(-0.1, 0.1, size=self.nr_aux)

        if self.missing_mode == "debug":
            # define the probability for each mode to occur, then sample mode first in each iteration
            # not realisitc, but useful for checking the implementations of the modes etc.
            self.mode_probs = [
                0,  # 0.2,
                0.5,  # 0.12,
                0,  # 0.12,
                0,  # 0.12,
                0,  # 0.12,
                0.5,  # 0.12,
                0,  # 0.1,
                0,
            ]  # 0.1
            assert sum(self.mode_probs) == 1
            self.aux_missing_p = [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5]
        elif self.missing_mode == "mcar":
            # `missing completely at random` - rarely true in practice
            # define missing probability of each feature
            self.cont_missing_p = [
                0.002,
                0.003,
                0.084,
                0.080,
                0.014,
                0.132,
                0.128,
                0.159,
                0.166,
                0.195,
            ]
            self.cat_missing_p = [
                0.004,
                0.010,
                0.015,
                0.061,
                0.264,
                0.018,
                0.009,
                0.017,
                0.317,
                0.023,
                0.005,
                0.006,
                0.087,
                0.001,
                0.047,
            ]
            self.aux_missing_p = [
                0.072,
                0.072,
                0.072,
                0.228,
                0.070,
                0.063,
                0.118,
                0.074,
                0.073,
                0.068,
                0.071,
                0.487,
                0.413,
                0.509,
                0.300,
            ]
            self.image_missing_p = 0.49
            if self.missing_factor is not None:
                self.cont_missing_p = [
                    min(1.0, el * self.missing_factor) for el in self.cont_missing_p
                ]
                self.cat_missing_p = [
                    min(1.0, el * self.missing_factor) for el in self.cat_missing_p
                ]
                self.aux_missing_p = [
                    min(1.0, el * self.missing_factor) for el in self.aux_missing_p
                ]
                self.image_missing_p = min(
                    1.0, self.image_missing_p * self.missing_factor
                )
        elif self.missing_mode == "emulate_tavi":
            # we simply sample from the missing patterns occuring in the TAVI dataset
            # this is the most realistic scenario
            dirname = os.path.dirname(__file__)
            self.cont_missing_masks = np.load(
                os.path.join(dirname, "tavi_statistics", "cont_missing_masks.npy")
            )
            self.cat_missing_masks = np.load(
                os.path.join(dirname, "tavi_statistics", "cat_missing_masks.npy")
            )
            self.aux_missing_masks = np.load(
                os.path.join(dirname, "tavi_statistics", "aux_missing_masks.npy")
            )
            self.image_missing_masks = np.load(
                os.path.join(dirname, "tavi_statistics", "image_missing_masks.npy")
            )
            self.mask_length = len(self.image_missing_masks)

        if not self.infinite_flag:
            cont_features = []
            cat_features = []
            aux_features = []
            img_features = []
            labels = []
            modes = []
            for _ in range(self.len_dataset):
                cont, cat, f_img, J, y = self._sample_datapoint()
                cont_mask, cat_mask, aux_mask, image_missing, mode = (
                    self._sample_masks()
                )
                cont[cont_mask] = np.nan
                cat[cat_mask] = np.nan
                J[aux_mask] = np.nan
                if image_missing:
                    f_img.fill(np.nan)
                cont_features.append(cont)
                cat_features.append(cat)
                img_features.append(f_img)
                aux_features.append(J)
                labels.append(y)
                modes.append(mode)
            self.cont_features = np.array(cont_features)
            self.cat_features = np.array(cat_features)
            self.img_features = np.array(img_features)
            self.aux_features = np.array(aux_features)
            self.labels = np.array(labels)
            self.modes = np.array(modes)

    def __getitem__(self, index):
        sample = {}

        if self.infinite_flag:
            # generate random datapoint online --> infinite dataset size
            cont, cat, f_img, J, y = self._sample_datapoint()
            cont_mask, cat_mask, aux_mask, image_missing, mode = self._sample_masks()
            cont[cont_mask] = np.nan
            cat[cat_mask] = np.nan
            J[aux_mask] = np.nan
            if image_missing:
                f_img.fill(np.nan)

        else:
            cont = self.cont_features[index]
            cat = self.cat_features[index]
            J = self.aux_features[index]
            f_img = self.img_features[index]
            y = self.labels[index]
            mode = self.modes[index]

        sample["label"] = y.astype(np.float32)
        sample["cont_table"] = cont.astype(np.float32)
        sample["cat_table"] = cat.astype(np.float32)
        sample["cont_imtable"] = J.astype(np.float32)
        sample["image"] = f_img.astype(np.float32)
        sample["modes"] = mode

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return self.len_dataset

    def _sample_datapoint(self):
        # sample the datapoint
        cont = np.random.randn(self.nr_cont_features).astype(np.float32)
        cat = (
            np.random.binomial(
                n=1, p=self.cat_table_prior, size=self.nr_cat_features
            ).astype(np.float32)
            * 2
            - 1.0
        )
        A = np.concatenate((cont, cat), axis=0)
        f_img = (
            np.sqrt(np.exp(self.log_var_I)) * np.random.randn(self.nr_image_features)
            + np.transpose(self.beta) @ A
        )
        J = (
            np.sqrt(np.exp(self.log_var_J)) * np.random.randn(self.nr_aux)
            + np.transpose(self.phi) @ f_img
        )
        logit_y = (
            np.transpose(self.alpha_I) @ (f_img - np.transpose(self.beta) @ A)
            + np.transpose(self.alpha_A) @ A
            + self.b_Y
        )
        y = self._boltzmann_sample(logit_y, 1.0)
        return cont, cat, f_img, J, y

    @staticmethod
    def _boltzmann_sample(logit, temperature):
        # we have only 2 states, that makes things easy
        # sigmoid with temperature
        p = np.exp(-np.logaddexp(0, -logit / temperature))
        if np.random.rand(1) < p:
            return np.array([1])
        else:
            return np.array([0])

    def _sample_masks(self):
        # add missing values
        if self.missing_mode == "debug":
            mode = np.random.choice(8, size=1, p=self.mode_probs)
            if mode in [1, 4, 5, 7]:
                # have at most 8 missing entries
                num_missing = np.random.randint(1, min(8, self.nr_cont_features + 1))
                cont_missing_mask = np.random.choice(
                    self.nr_cont_features, size=num_missing, replace=False
                )
            else:
                cont_missing_mask = [False] * self.nr_cont_features
            if mode in [2, 4, 6, 7]:
                # have at most 8 missing entries
                num_missing = np.random.randint(1, min(8, self.nr_cat_features + 1))
                cat_missing_mask = np.random.choice(
                    self.nr_cat_features, size=num_missing, replace=False
                )
            else:
                cat_missing_mask = [False] * self.nr_cat_features
            if mode in [3, 5, 6, 7]:
                image_missing = True
            else:
                image_missing = False
            aux_missing_mask = np.random.rand(self.nr_aux) < self.aux_missing_p
        elif self.missing_mode == "mcar":
            cont_missing_mask = (
                np.random.rand(self.nr_cont_features) < self.cont_missing_p
            )
            cat_missing_mask = np.random.rand(self.nr_cat_features) < self.cat_missing_p
            aux_missing_mask = np.random.rand(self.nr_aux) < self.aux_missing_p
            image_missing = np.random.rand(1) < self.image_missing_p
            has_missing_cont = cont_missing_mask.any()
            has_missing_cat = cat_missing_mask.any()
            mode = self._get_mode(has_missing_cont, has_missing_cat, image_missing)
        elif self.missing_mode == "emulate_tavi":
            idx = np.random.randint(0, self.mask_length)
            cont_missing_mask = self.cont_missing_masks[idx]
            cat_missing_mask = self.cat_missing_masks[idx]
            aux_missing_mask = self.aux_missing_masks[idx]
            image_missing = self.image_missing_masks[idx]
            has_missing_cont = cont_missing_mask.any()
            has_missing_cat = cat_missing_mask.any()
            mode = self._get_mode(has_missing_cont, has_missing_cat, image_missing)
        return (
            cont_missing_mask,
            cat_missing_mask,
            aux_missing_mask,
            image_missing,
            mode,
        )

    @staticmethod
    def _get_mode(has_cont_missing, has_cat_missing, img_missing):
        if not img_missing and not has_cont_missing and not has_cat_missing:
            return 0
        elif not img_missing and has_cont_missing and not has_cat_missing:
            return 1
        elif not img_missing and not has_cont_missing and has_cat_missing:
            return 2
        elif img_missing and not has_cont_missing and not has_cat_missing:
            return 3
        elif not img_missing and has_cont_missing and has_cat_missing:
            return 4
        elif img_missing and has_cont_missing and not has_cat_missing:
            return 5
        elif img_missing and not has_cont_missing and has_cat_missing:
            return 6
        elif img_missing and has_cont_missing and has_cat_missing:
            return 7
