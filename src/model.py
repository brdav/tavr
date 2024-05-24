import math
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchmetrics

from .landmark_localizer import CombinedLandmarkLocalizer
from .utils import (
    TABULAR_NAMES,
    MEASUREMENT_NAMES,
    InverseSquareRootLR,
    align_sitk_image,
    extract_roi,
)

model_urls = {
    "ResNet": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/resolve/main/resnet_50_23dataset.pth",
    "SwinTransformer": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt",
}


class ProbabilisticModel(pl.LightningModule):
    def __init__(
        self,
        neural_network: nn.Module,
        nr_image_features: int = 16,
        nr_cont_table_features: int = 10,
        nr_cat_table_features: int = 15,
        nr_aux_features: int = 15,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        use_mtl: bool = True,
        freeze_bn: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["neural_network"])

        #
        # MODEL
        #
        self.nr_image_feats = nr_image_features
        self.nr_cont_table_feats = nr_cont_table_features
        self.nr_cat_table_feats = nr_cat_table_features
        self.nr_aux_feats = nr_aux_features
        self.neural_network = neural_network
        self.use_mtl = use_mtl
        self.freeze_bn = freeze_bn
        if self.freeze_bn:
            for m in self.neural_network.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    for param in m.parameters():
                        param.requires_grad = False
        # using groups here to keep DOF / parameter count low
        if self.neural_network.__class__.__name__ == "ResNet":
            self.reduce_im = nn.Conv3d(
                2048,
                self.nr_image_feats,
                kernel_size=1,
                groups=self.nr_image_feats,
                bias=False,
            )
        elif self.neural_network.__class__.__name__ == "SwinTransformer":
            self.reduce_im = nn.Conv3d(
                768,
                self.nr_image_feats,
                kernel_size=1,
                groups=self.nr_image_feats,
                bias=False,
            )
        else:
            raise ValueError
        self.alpha_A = nn.Parameter(
            torch.empty(self.nr_cont_table_feats + self.nr_cat_table_feats, 1)
        )
        self.alpha_I = nn.Parameter(torch.empty(self.nr_image_feats, 1))
        self.beta = nn.Parameter(
            torch.empty(
                self.nr_cont_table_feats + self.nr_cat_table_feats, self.nr_image_feats
            )
        )
        self.b_Y = nn.Parameter(torch.empty(1, 1))
        self.log_var_I = nn.Parameter(torch.empty(1))
        if self.use_mtl:
            self.phi = nn.Parameter(torch.empty(self.nr_image_feats, self.nr_aux_feats))
            self.log_var_J = nn.Parameter(torch.empty(self.nr_aux_feats))
        # landmark localization model
        self.landmark_localizer = CombinedLandmarkLocalizer()

        # register buffers here, fill them later
        self.register_buffer("cat_table_prior", torch.zeros(self.nr_cat_table_feats))
        self.register_buffer("cont_mean", torch.zeros([1, self.nr_cont_table_feats]))
        self.register_buffer("cont_std", torch.ones([1, self.nr_cont_table_feats]))
        self.register_buffer("aux_mean", torch.zeros([1, self.nr_aux_feats]))
        self.register_buffer("aux_std", torch.ones([1, self.nr_aux_feats]))

        #
        # OTHER HYPERPARAMETERS
        #
        self.lr = lr
        self.weight_decay = weight_decay

        self.val_metric = torchmetrics.classification.BinaryAUROC()
        self.test_metric = torchmetrics.classification.BinaryAUROC()
        self.init_weights()

    def init_weights(self):
        if self.neural_network.__class__.__name__ == "ResNet":
            checkpoint = torch.hub.load_state_dict_from_url(
                model_urls["ResNet"],
                progress=True,
                map_location=lambda storage, loc: storage,
            )
            state_dict = {
                k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
            }
            self.neural_network.load_state_dict(state_dict, strict=True)
        elif self.neural_network.__class__.__name__ == "SwinTransformer":
            checkpoint = torch.hub.load_state_dict_from_url(
                model_urls["SwinTransformer"],
                progress=True,
                map_location=lambda storage, loc: storage,
            )
            state_dict = {
                k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
            }
            ignore_keys = [
                "norm.weight",
                "norm.bias",
                "convTrans3d.weight",
                "convTrans3d.bias",
                "rotation_head.weight",
                "rotation_head.bias",
                "contrastive_head.weight",
                "contrastive_head.bias",
            ]
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in ignore_keys:
                    continue
                if "mlp.fc1." in k:
                    new_k = k.replace("mlp.fc1.", "mlp.linear1.")
                elif "mlp.fc2." in k:
                    new_k = k.replace("mlp.fc2.", "mlp.linear2.")
                else:
                    new_k = k
                filtered_state_dict[new_k] = v
            self.neural_network.load_state_dict(filtered_state_dict, strict=True)
        else:
            raise ValueError

        nn.init.normal_(
            self.reduce_im.weight, mean=0, std=math.sqrt(self.nr_image_feats / 2048)
        )
        nn.init.normal_(
            self.alpha_A,
            mean=0,
            std=0.01 / math.sqrt(self.nr_cont_table_feats + self.nr_cat_table_feats),
        )
        nn.init.normal_(self.alpha_I, mean=0, std=0.01 / math.sqrt(self.nr_image_feats))
        nn.init.normal_(
            self.beta,
            mean=0,
            std=0.01 / math.sqrt(self.nr_cont_table_feats + self.nr_cat_table_feats),
        )
        nn.init.constant_(self.b_Y, val=0)
        nn.init.constant_(self.log_var_I, val=0)
        if self.use_mtl:
            nn.init.normal_(self.phi, mean=0, std=0.01 / math.sqrt(self.nr_image_feats))
            nn.init.constant_(self.log_var_J, val=0)

    def setup(self, stage):
        if stage == "fit":
            self.cat_table_prior = torch.from_numpy(
                self.trainer.datamodule.trainset.cat_table_prior
            )
            self.cont_mean = torch.from_numpy(self.trainer.datamodule.trainset.mean)
            self.cont_std = torch.from_numpy(self.trainer.datamodule.trainset.std)
            self.aux_mean = torch.from_numpy(self.trainer.datamodule.trainset.aux_mean)
            self.aux_std = torch.from_numpy(self.trainer.datamodule.trainset.aux_std)

    def forward(
        self,
        image: Optional[sitk.Image] = None,
        tabular: Optional[pd.Series] = None,
        measurements: Optional[pd.Series] = None,
    ):
        """full prediction"""

        if image is not None:
            image = align_sitk_image(image)
            # landmark localization
            coords = self.landmark_localizer(image)
            # ROI extraction
            roi = extract_roi(image, coords)
            # preprocessing
            roi_pt = torch.from_numpy(roi).to(self.device)
            roi_pt = (roi_pt - roi_pt.mean()) / roi_pt.std()
        else:
            roi = None
            roi_pt = None

        if tabular is not None:
            # preprocessing
            tabular_pt = torch.tensor(
                [tabular[name] for name in TABULAR_NAMES],
                dtype=torch.float,
                device=self.device,
            )
            cont, cat = torch.split(tabular_pt, (10, 15))
            cont = ((cont - self.cont_mean) / self.cont_std).squeeze(0)
            cat = cat * 2.0 - 1.0
        else:
            # in this case, all tabular variables need to be marginalized over
            cont = torch.full(
                [self.nr_cont_table_feats],
                np.nan,
                dtype=torch.float,
                device=self.device,
            )
            cat = torch.full(
                [self.nr_cat_table_feats], np.nan, dtype=torch.float, device=self.device
            )

        #
        # PROBABILISTIC MODEL
        #
        mode = self.get_mode(cont, cat, roi_pt)
        if mode in [0, 1, 2, 4]:
            # image is present
            dnn_out = self.neural_network(roi_pt.unsqueeze(0).unsqueeze(0))
            if isinstance(dnn_out, list):
                dnn_out = dnn_out[-1]
            if dnn_out.dim() == 5:
                dnn_out = nn.functional.adaptive_avg_pool3d(dnn_out, (1, 1, 1))
            else:
                dnn_out = dnn_out.view(dnn_out.shape[0], dnn_out.shape[1], 1, 1, 1)
            img = self.reduce_im(dnn_out).view(-1, 1)
            aux = torch.full(
                [self.nr_aux_feats], np.nan, dtype=torch.float, device=self.device
            )  # dummy
        else:
            # image is missing, try to marginalize using image measurements
            img = None
            if measurements is not None:
                aux = torch.tensor(
                    [measurements[name] for name in MEASUREMENT_NAMES],
                    dtype=torch.float,
                    device=self.device,
                )
                aux = ((aux - self.aux_mean) / self.aux_std).squeeze(0)
            else:
                aux = torch.full(
                    [self.nr_aux_feats],
                    np.nan,
                    dtype=torch.float,
                    device=self.device,
                )  # dummy

        target = torch.zeros((1, 1), dtype=torch.float, device=self.device)  # dummy

        cont = cont.unsqueeze(1)
        cat = cat.unsqueeze(1)
        aux = aux.unsqueeze(1)
        _, mu_y = self.forward_probabilistic(img, cont, cat, aux, target, mode)
        return mu_y, roi

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward_batch(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward_batch(batch)
        self.val_metric(preds, batch["label"].long())
        self.log("val_loss", loss)
        self.log("val_AUROC", self.val_metric)

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward_batch(batch)
        self.test_metric(preds, batch["label"].long())
        self.log("test_loss", loss)
        self.log("test_AUROC", self.test_metric)

    def forward_batch(self, batch):
        images = batch["image"]
        cont_table = batch["cont_table"]
        cat_table = batch["cat_table"]
        cont_aux = batch["cont_imtable"]
        modes = batch["modes"]
        targets = batch["label"]

        # check via shape of `images` whether we need to run through DNN or not
        if torch.numel(images[0]) == self.nr_image_feats:
            img_features = images
        else:
            # only forward valid images through DNN
            has_image = (modes == 0) | (modes == 1) | (modes == 2) | (modes == 4)
            img_features = torch.zeros(
                targets.size(0),
                self.nr_image_feats,
                dtype=torch.float,
                device=self.device,
            )

            if has_image.any():
                dnn_out = self.neural_network(images[has_image])
                if isinstance(dnn_out, list):
                    dnn_out = dnn_out[-1]
                if dnn_out.dim() == 5:
                    dnn_out = nn.functional.adaptive_avg_pool3d(dnn_out, (1, 1, 1))
                else:
                    dnn_out = dnn_out.view(dnn_out.shape[0], dnn_out.shape[1], 1, 1, 1)
                img_features[has_image] = self.reduce_im(dnn_out).view(
                    dnn_out.size(0), -1
                )

        loss_batch = 0
        mu_y_batch = []
        # during inference, `aux_cont_feats` will just be all NaNs
        for img, cont, cat, aux_cont, target, m in zip(
            img_features, cont_table, cat_table, cont_aux, targets, modes
        ):
            # add dim for calculations in forward
            img = img.unsqueeze(1)
            cont = cont.unsqueeze(1)
            cat = cat.unsqueeze(1)
            aux_cont = aux_cont.unsqueeze(1)
            target = target.unsqueeze(1)
            loss, mu_y = self.forward_probabilistic(img, cont, cat, aux_cont, target, m)
            loss_batch += loss
            mu_y_batch.append(mu_y)
        loss_batch /= targets.size(0)
        return loss_batch, torch.cat(mu_y_batch)

    def forward_probabilistic(self, img, cont, cat, aux_cont, target, mode):
        if mode == 0:
            loss, mu_y = self.forward_complete(img, cont, cat, aux_cont, target)
        elif mode == 1:
            loss, mu_y = self.forward_missing_cont(img, cont, cat, aux_cont, target)
        elif mode == 2:
            loss, mu_y = self.forward_missing_cat(img, cont, cat, aux_cont, target)
        elif mode == 3:
            loss, mu_y = self.forward_missing_img(cont, cat, aux_cont, target)
        elif mode == 4:
            loss, mu_y = self.forward_missing_cont_cat(img, cont, cat, aux_cont, target)
        elif mode == 5:
            loss, mu_y = self.forward_missing_cont_img(cont, cat, aux_cont, target)
        elif mode == 6:
            loss, mu_y = self.forward_missing_cat_img(cont, cat, aux_cont, target)
        elif mode == 7:
            loss, mu_y = self.forward_missing_cont_cat_img(cont, cat, aux_cont, target)
        return loss, mu_y

    def forward_complete(self, f_img, cont_table, cat_table, cont_aux, target):
        A = torch.cat((cont_table, cat_table), dim=0)
        mu_I = self.beta.t() @ A
        cov_I = torch.diag(torch.exp(self.log_var_I).repeat(self.nr_image_feats))
        logits_y = self.alpha_I.t() @ (f_img - mu_I) + self.alpha_A.t() @ A + self.b_Y

        loss = self.boltzmann_nll(logits_y, target)
        loss += self.gaussian_nll(f_img, mu_I, cov_I)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if valid_J_mask.any() and self.use_mtl:
            valid_phi = self.phi[:, valid_J_mask]
            mu_J = valid_phi.t() @ f_img
            cov_J = torch.diag(torch.exp(self.log_var_J[valid_J_mask]))
            loss += self.gaussian_nll(cont_aux[valid_J_mask], mu_J, cov_J)

        return loss, torch.sigmoid(logits_y).detach()

    def forward_missing_cont(self, f_img, cont_table, cat_table, cont_aux, target):
        # Remember that we standardized the input features (zero mean, unit variance), therefore:
        # - mu_A is a vector with zeros for missing features and the actual value for the present features
        # - cov_A is a diagonal matrix which is one for the missing features and 0 for the present features
        all_feats = torch.cat((cont_table, cat_table), dim=0)
        mu_A = all_feats.clone()
        mu_A[mu_A != mu_A] = 0.0
        cov_A = all_feats.isnan().float().squeeze(1)
        cov_A = torch.diag(cov_A)

        mu_I = self.beta.t() @ mu_A
        cov_I = (
            torch.diag(torch.exp(self.log_var_I).repeat(self.nr_image_feats))
            + self.beta.t() @ cov_A @ self.beta
        )

        num = (
            torch.exp(self.log_var_I) * self.alpha_I.t()
            + self.alpha_A.t() @ cov_A @ self.beta
        )
        inv_denom = torch.inverse(cov_I)

        mu_p = self.alpha_A.t() @ mu_A + num @ inv_denom @ (f_img - mu_I) + self.b_Y
        var_p = (
            self.alpha_A.t() @ cov_A @ self.alpha_A
            + torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I
            - num @ inv_denom @ num.t()
        )

        logits_y = mu_p / (1.0 + math.pi / 8.0 * var_p) ** 0.5

        loss = self.boltzmann_nll(logits_y, target)
        loss += self.gaussian_nll(f_img, mu_I, cov_I)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if valid_J_mask.any() and self.use_mtl:
            valid_phi = self.phi[:, valid_J_mask]
            mu_J = valid_phi.t() @ f_img
            cov_J = torch.diag(torch.exp(self.log_var_J[valid_J_mask]))
            loss += self.gaussian_nll(cont_aux[valid_J_mask], mu_J, cov_J)

        return loss, torch.sigmoid(logits_y).detach()

    def forward_missing_cat(self, f_img, cont_table, cat_table, cont_aux, target):
        nan_idx = cat_table.isnan().nonzero(as_tuple=True)
        fill_range = torch.tensor([-1, 1], dtype=cat_table.dtype, device=self.device)
        fill_values = torch.stack(
            [
                g.flatten()
                for g in torch.meshgrid([fill_range] * len(nan_idx[0]), indexing="ij")
            ],
            dim=1,
        )

        cov_I_v = torch.diag(torch.exp(self.log_var_I).repeat(self.nr_image_feats))
        mu_y = 0
        ll = []
        for val in fill_values:
            cat_table_v = torch.index_put(cat_table, nan_idx, val)
            ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
            A_v = torch.cat((cont_table, cat_table_v), dim=0)
            mu_I_v = self.beta.t() @ A_v
            logits_y_v = (
                self.alpha_I.t() @ (f_img - mu_I_v) + self.alpha_A.t() @ A_v + self.b_Y
            )
            ll_y_v = -self.boltzmann_nll(logits_y_v, target)
            ll_I_v = -self.gaussian_nll(f_img, mu_I_v, cov_I_v)
            ll.append(ll_A_v + ll_y_v + ll_I_v)

            mu_y += (torch.sigmoid(logits_y_v) * torch.exp(ll_A_v)).detach()

        loss = -torch.logsumexp(torch.stack(ll, dim=0), dim=0)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if valid_J_mask.any() and self.use_mtl:
            valid_phi = self.phi[:, valid_J_mask]
            mu_J = valid_phi.t() @ f_img
            cov_J = torch.diag(torch.exp(self.log_var_J[valid_J_mask]))
            loss += self.gaussian_nll(cont_aux[valid_J_mask], mu_J, cov_J)

        return loss, mu_y

    def forward_missing_img(self, cont_table, cat_table, cont_aux, target):
        A = torch.cat((cont_table, cat_table), dim=0)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if (not valid_J_mask.any()) or not self.use_mtl:
            mu_p = self.alpha_A.t() @ A + self.b_Y
            var_p = torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I

            logits_y = mu_p / (1.0 + math.pi / 8.0 * var_p) ** 0.5
            loss = self.boltzmann_nll(logits_y, target)

        else:
            valid_phi = self.phi[:, valid_J_mask]
            mu_J = valid_phi.t() @ self.beta.t() @ A
            cov_J = (
                torch.diag(torch.exp(self.log_var_J[valid_J_mask]))
                + torch.exp(self.log_var_I) * valid_phi.t() @ valid_phi
            )
            num = torch.exp(self.log_var_I) * self.alpha_I.t() @ valid_phi
            inv_denom = torch.inverse(cov_J)
            mu_p = (
                self.alpha_A.t() @ A
                + num @ inv_denom @ (cont_aux[valid_J_mask] - mu_J)
                + self.b_Y
            )
            var_p = (
                torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I
                - num @ inv_denom @ num.t()
            )

            logits_y = mu_p / (1.0 + math.pi / 8.0 * var_p) ** 0.5
            loss = self.boltzmann_nll(logits_y, target)
            loss += self.gaussian_nll(cont_aux[valid_J_mask], mu_J, cov_J)

        return loss, torch.sigmoid(logits_y).detach()

    def forward_missing_cont_cat(self, f_img, cont_table, cat_table, cont_aux, target):
        nan_idx = cat_table.isnan().nonzero(as_tuple=True)
        fill_range = torch.tensor([-1, 1], dtype=cat_table.dtype, device=self.device)
        fill_values = torch.stack(
            [
                g.flatten()
                for g in torch.meshgrid([fill_range] * len(nan_idx[0]), indexing="ij")
            ],
            dim=1,
        )

        mu_y = 0
        ll = []
        for val in fill_values:
            cat_table_v = torch.index_put(cat_table, nan_idx, val)
            ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
            A_v = torch.cat((cont_table, cat_table_v), dim=0)

            mu_A_v = A_v.clone()
            mu_A_v[mu_A_v != mu_A_v] = 0.0
            cov_A_v = A_v.isnan().float().squeeze(1)
            cov_A_v = torch.diag(cov_A_v)

            mu_I_v = self.beta.t() @ mu_A_v
            cov_I_v = (
                torch.diag(torch.exp(self.log_var_I).repeat(self.nr_image_feats))
                + self.beta.t() @ cov_A_v @ self.beta
            )

            num = (
                torch.exp(self.log_var_I) * self.alpha_I.t()
                + self.alpha_A.t() @ cov_A_v @ self.beta
            )
            inv_denom = torch.inverse(cov_I_v)

            mu_p_v = (
                self.alpha_A.t() @ mu_A_v
                + num @ inv_denom @ (f_img - mu_I_v)
                + self.b_Y
            )
            var_p_v = (
                self.alpha_A.t() @ cov_A_v @ self.alpha_A
                + torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I
                - num @ inv_denom @ num.t()
            )

            logits_y_v = mu_p_v / (1.0 + math.pi / 8.0 * var_p_v) ** 0.5

            ll_y_v = -self.boltzmann_nll(logits_y_v, target)
            ll_I_v = -self.gaussian_nll(f_img, mu_I_v, cov_I_v)
            ll.append(ll_A_v + ll_y_v + ll_I_v)

            mu_y += (torch.sigmoid(logits_y_v) * torch.exp(ll_A_v)).detach()

        loss = -torch.logsumexp(torch.stack(ll, dim=0), dim=0)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if valid_J_mask.any() and self.use_mtl:
            valid_phi = self.phi[:, valid_J_mask]
            mu_J = valid_phi.t() @ f_img
            cov_J = torch.diag(torch.exp(self.log_var_J[valid_J_mask]))
            loss += self.gaussian_nll(cont_aux[valid_J_mask], mu_J, cov_J)

        return loss, mu_y

    def forward_missing_cont_img(self, cont_table, cat_table, cont_aux, target):
        all_feats = torch.cat((cont_table, cat_table), dim=0)
        mu_A = all_feats.clone()
        mu_A[mu_A != mu_A] = 0.0
        cov_A = all_feats.isnan().float().squeeze(1)
        cov_A = torch.diag(cov_A)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if valid_J_mask.any() and self.use_mtl:

            valid_phi = self.phi[:, valid_J_mask]
            mu_J = valid_phi.t() @ self.beta.t() @ mu_A
            cov_J = valid_phi.t() @ (
                torch.diag(torch.exp(self.log_var_I).repeat(self.nr_image_feats))
                + self.beta.t() @ cov_A @ self.beta
            ) @ valid_phi + torch.diag(torch.exp(self.log_var_J[valid_J_mask]))
            num = (
                torch.exp(self.log_var_I) * self.alpha_I.t()
                + self.alpha_A.t() @ cov_A @ self.beta
            ) @ valid_phi
            inv_denom = torch.inverse(cov_J)

            mu_p = (
                self.alpha_A.t() @ mu_A
                + num @ inv_denom @ (cont_aux[valid_J_mask] - mu_J)
                + self.b_Y
            )
            var_p = (
                self.alpha_A.t() @ cov_A @ self.alpha_A
                + torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I
                - num @ inv_denom @ num.t()
            )

            logits_y = mu_p / (1.0 + math.pi / 8.0 * var_p) ** 0.5

            loss = self.boltzmann_nll(logits_y, target)
            loss += self.gaussian_nll(cont_aux[valid_J_mask], mu_J, cov_J)

        else:
            mu_p = self.alpha_A.t() @ mu_A + self.b_Y
            var_p = (
                self.alpha_A.t() @ cov_A @ self.alpha_A
                + torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I
            )

            logits_y = mu_p / (1.0 + math.pi / 8.0 * var_p) ** 0.5
            loss = self.boltzmann_nll(logits_y, target)

        return loss, torch.sigmoid(logits_y).detach()

    def forward_missing_cat_img(self, cont_table, cat_table, cont_aux, target):
        nan_idx = cat_table.isnan().nonzero(as_tuple=True)
        fill_range = torch.tensor([-1, 1], dtype=cat_table.dtype, device=self.device)
        fill_values = torch.stack(
            [
                g.flatten()
                for g in torch.meshgrid([fill_range] * len(nan_idx[0]), indexing="ij")
            ],
            dim=1,
        )

        valid_J_mask = ~cont_aux.isnan().squeeze(1)

        mu_y = 0
        ll = []
        if (not valid_J_mask.any()) or not self.use_mtl:
            for val in fill_values:
                cat_table_v = torch.index_put(cat_table, nan_idx, val)
                ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
                A_v = torch.cat((cont_table, cat_table_v), dim=0)
                mu_p_v = self.alpha_A.t() @ A_v + self.b_Y
                var_p_v = torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I
                logits_y_v = mu_p_v / (1.0 + math.pi / 8.0 * var_p_v) ** 0.5

                ll_y_v = -self.boltzmann_nll(logits_y_v, target)
                ll.append(ll_A_v + ll_y_v)

                mu_y += (torch.sigmoid(logits_y_v) * torch.exp(ll_A_v)).detach()

        else:
            valid_phi = self.phi[:, valid_J_mask]
            for val in fill_values:
                cat_table_v = torch.index_put(cat_table, nan_idx, val)
                ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
                A_v = torch.cat((cont_table, cat_table_v), dim=0)
                mu_J_v = valid_phi.t() @ self.beta.t() @ A_v
                cov_J_v = (
                    torch.diag(torch.exp(self.log_var_J[valid_J_mask]))
                    + torch.exp(self.log_var_I) * valid_phi.t() @ valid_phi
                )
                num = torch.exp(self.log_var_I) * self.alpha_I.t() @ valid_phi
                inv_denom = torch.inverse(cov_J_v)
                mu_p_v = (
                    self.alpha_A.t() @ A_v
                    + num @ inv_denom @ (cont_aux[valid_J_mask] - mu_J_v)
                    + self.b_Y
                )
                var_p_v = (
                    torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I
                    - num @ inv_denom @ num.t()
                )

                logits_y_v = mu_p_v / (1.0 + math.pi / 8.0 * var_p_v) ** 0.5

                ll_y_v = -self.boltzmann_nll(logits_y_v, target)
                ll_J_v = -self.gaussian_nll(cont_aux[valid_J_mask], mu_J_v, cov_J_v)
                ll.append(ll_A_v + ll_y_v + ll_J_v)

                mu_y += (torch.sigmoid(logits_y_v) * torch.exp(ll_A_v)).detach()

        loss = -torch.logsumexp(torch.stack(ll, dim=0), dim=0)

        return loss, mu_y

    def forward_missing_cont_cat_img(self, cont_table, cat_table, cont_aux, target):
        nan_idx = cat_table.isnan().nonzero(as_tuple=True)
        fill_range = torch.tensor([-1, 1], dtype=cat_table.dtype, device=self.device)
        fill_values = torch.stack(
            [
                g.flatten()
                for g in torch.meshgrid([fill_range] * len(nan_idx[0]), indexing="ij")
            ],
            dim=1,
        )

        valid_J_mask = ~cont_aux.isnan().squeeze(1)

        mu_y = 0
        ll = []
        if (not valid_J_mask.any()) or not self.use_mtl:
            for val in fill_values:
                cat_table_v = torch.index_put(cat_table, nan_idx, val)
                ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
                A_v = torch.cat((cont_table, cat_table_v), dim=0)

                mu_A_v = A_v.clone()
                mu_A_v[mu_A_v != mu_A_v] = 0.0
                cov_A_v = A_v.isnan().float().squeeze(1)
                cov_A_v = torch.diag(cov_A_v)

                mu_p_v = self.alpha_A.t() @ mu_A_v + self.b_Y
                var_p_v = (
                    self.alpha_A.t() @ cov_A_v @ self.alpha_A
                    + torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I
                )

                logits_y_v = mu_p_v / (1.0 + math.pi / 8.0 * var_p_v) ** 0.5

                ll_y_v = -self.boltzmann_nll(logits_y_v, target)
                ll.append(ll_A_v + ll_y_v)

                mu_y += (torch.sigmoid(logits_y_v) * torch.exp(ll_A_v)).detach()

        else:
            valid_phi = self.phi[:, valid_J_mask]
            for val in fill_values:
                cat_table_v = torch.index_put(cat_table, nan_idx, val)
                ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
                A_v = torch.cat((cont_table, cat_table_v), dim=0)

                mu_A_v = A_v.clone()
                mu_A_v[mu_A_v != mu_A_v] = 0.0
                cov_A_v = A_v.isnan().float().squeeze(1)
                cov_A_v = torch.diag(cov_A_v)

                mu_J_v = valid_phi.t() @ self.beta.t() @ mu_A_v
                cov_J_v = valid_phi.t() @ (
                    torch.diag(torch.exp(self.log_var_I).repeat(self.nr_image_feats))
                    + self.beta.t() @ cov_A_v @ self.beta
                ) @ valid_phi + torch.diag(torch.exp(self.log_var_J[valid_J_mask]))

                num = (
                    torch.exp(self.log_var_I) * self.alpha_I.t()
                    + self.alpha_A.t() @ cov_A_v @ self.beta
                ) @ valid_phi
                inv_denom = torch.inverse(cov_J_v)

                mu_p_v = (
                    self.alpha_A.t() @ mu_A_v
                    + num @ inv_denom @ (cont_aux[valid_J_mask] - mu_J_v)
                    + self.b_Y
                )
                var_p_v = (
                    self.alpha_A.t() @ cov_A_v @ self.alpha_A
                    + torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I
                    - num @ inv_denom @ num.t()
                )

                logits_y_v = mu_p_v / (1.0 + math.pi / 8.0 * var_p_v) ** 0.5

                ll_y_v = -self.boltzmann_nll(logits_y_v, target)
                ll_J_v = -self.gaussian_nll(cont_aux[valid_J_mask], mu_J_v, cov_J_v)
                ll.append(ll_A_v + ll_y_v + ll_J_v)

                mu_y += (torch.sigmoid(logits_y_v) * torch.exp(ll_A_v)).detach()

        loss = -torch.logsumexp(torch.stack(ll, dim=0), dim=0)

        return loss, mu_y

    def get_cat_joint_logprob(self, sampled_vals, indices):
        # we have a Bernoulli distribution
        ks = (sampled_vals + 1.0) / 2.0  # to range {0, 1}
        ps = self.cat_table_prior[indices]
        log_probs = ks * torch.log(ps) + (1 - ks) * torch.log(1 - ps)  # Bernoulli
        return torch.sum(log_probs)

    def boltzmann_nll(self, logit, target, temperature=1):
        # negative log-likelihood
        return nn.functional.binary_cross_entropy_with_logits(
            logit / temperature, target, reduction="none"
        )

    def gaussian_nll(self, value, mu, cov_matrix):
        # negative log-likelihood
        sigma_matrix = torch.linalg.cholesky(cov_matrix)
        diff = (value - mu).squeeze(1)
        M = torch.distributions.multivariate_normal._batch_mahalanobis(
            sigma_matrix, diff
        )
        half_log_det = sigma_matrix.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        # the constant term (no gradient) is included for now
        return 0.5 * (mu.shape[0] * math.log(2 * math.pi) + M) + half_log_det

    def on_train_epoch_start(self):
        # this will prompt the dataset to randomly pick another version
        self.trainer.datamodule.trainset.image_dataset = None

    def train(self, mode=True):
        super().train(mode=mode)
        if self.freeze_bn:
            # extra treatment for BN layers
            for m in self.neural_network.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def classifier_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("dnn."):
                yield param

    def dnn_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("dnn."):
                yield param

    def configure_optimizers(self):
        param_groups = [
            {
                "name": "classifier",
                "params": filter(
                    lambda p: p.requires_grad, self.classifier_parameters()
                ),
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            }
        ]
        param_groups.append(
            {
                "name": "dnn",
                "params": filter(lambda p: p.requires_grad, self.dnn_parameters()),
                "lr": self.lr / 10.0,
                "weight_decay": self.weight_decay,
            }
        )
        optimizer = torch.optim.AdamW(params=param_groups)
        scheduler = {"scheduler": InverseSquareRootLR(optimizer), "interval": "epoch"}
        return [optimizer], [scheduler]

    @staticmethod
    def get_mode(cont, cat, roi):
        has_img_missing = roi == None
        has_cont_missing = torch.any(torch.isnan(cont))
        has_cat_missing = torch.any(torch.isnan(cat))
        if ~has_img_missing & ~has_cont_missing & ~has_cat_missing:  # all complete
            mode = 0
        elif ~has_img_missing & has_cont_missing & ~has_cat_missing:  # cont missing
            mode = 1
        elif ~has_img_missing & ~has_cont_missing & has_cat_missing:  # cat missing
            mode = 2
        elif has_img_missing & ~has_cont_missing & ~has_cat_missing:  # img missing
            mode = 3
        elif ~has_img_missing & has_cont_missing & has_cat_missing:  # cont cat missing
            mode = 4
        elif has_img_missing & has_cont_missing & ~has_cat_missing:  # cont img missing
            mode = 5
        elif has_img_missing & ~has_cont_missing & has_cat_missing:  # cat img missing
            mode = 6
        elif (
            has_img_missing & has_cont_missing & has_cat_missing
        ):  # cont cat img missing
            mode = 7
        return mode
