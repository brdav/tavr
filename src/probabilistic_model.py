import copy
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from src.model_helpers import resnet3d, utils_model


class ProbabilisticModel(pl.LightningModule):
    def __init__(self,
                 metrics_list: nn.ModuleList,  # datamodule
                 pos_class_weight: float,  # datamodule
                 nr_table_features: int,  # datamodule
                 nr_aux: int,  # datamodule
                 cat_table_prior: np.array,  # datamodule
                 gt_model: dict,  # datamodule
                 nr_image_features: int,  # datamodule
                 iterations: int = 10000,
                 warmup_steps: int = 10,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 use_lr_scheduler: bool = True,
                 use_tab: bool = True,
                 use_aux_as_input: bool = False,
                 use_aux_as_output: bool = True,
                 freeze_cnn_bn: bool = True,
                 fine_tune_cnn: bool = True):
        super().__init__()

        self.gt_model = gt_model
        self.iterations = iterations
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.valid_metrics_list = metrics_list
        self.test_metrics_list = copy.deepcopy(metrics_list)
        self.nr_image_feats = nr_image_features
        if use_aux_as_input and use_tab:
            self.nr_table_feats = nr_table_features + nr_aux
        elif use_tab:
            self.nr_table_feats = nr_table_features
        elif use_aux_as_input:
            self.nr_table_feats = nr_aux
        else:
            raise ValueError
        self.nr_aux_feats = nr_aux

        self.use_tab = use_tab
        self.use_aux_as_input = use_aux_as_input
        self.use_aux_as_output = use_aux_as_output

        # use them only one way
        if self.use_aux_as_output:
            assert not self.use_aux_as_input

        self.register_buffer('pos_weight', torch.tensor([pos_class_weight]))
        self.register_buffer(
            'cat_table_prior', torch.from_numpy(cat_table_prior))

        self.freeze_cnn_bn = freeze_cnn_bn
        self.fine_tune_cnn = fine_tune_cnn

        self.cnn = resnet3d.ResNet3D(load_pretrained='med3d')
        utils_model.set_freeze(
            self.cnn, freeze_bn=self.freeze_cnn_bn, freeze_other=not self.fine_tune_cnn)
        # using groups here to keep DOF / parameter count low
        self.reduce_im = nn.Conv3d(
            2048, self.nr_image_feats, kernel_size=1, groups=self.nr_image_feats, bias=False)

        # model parameters
        self.alpha_A = nn.Parameter(torch.empty(self.nr_table_feats, 1))
        self.alpha_I = nn.Parameter(torch.empty(self.nr_image_feats, 1))
        self.beta = nn.Parameter(torch.empty(
            self.nr_table_feats, self.nr_image_feats))
        self.b_Y = nn.Parameter(torch.empty(1, 1))
        self.log_var_I = nn.Parameter(torch.empty(1))

        if self.use_aux_as_output:
            self.phi = nn.Parameter(torch.empty(
                self.nr_image_feats, self.nr_aux_feats))
            self.log_var_J = nn.Parameter(torch.empty(self.nr_aux_feats))

        self.register_buffer('log_var_y', torch.tensor(
            0, dtype=torch.float))  # scalar

        self.init_weights()

        if self.gt_model is not None:
            self.register_buffer('gt_log_var_I', torch.tensor(
                [math.log(self.gt_model['sigma_I'] ** 2)]).to(torch.float))
            self.register_buffer('gt_log_var_J', torch.from_numpy(
                np.log(self.gt_model['sigma_J'] ** 2)).to(torch.float))
            self.register_buffer('gt_phi', torch.from_numpy(
                self.gt_model['phi']).to(torch.float))
            self.register_buffer('gt_b_Y', torch.tensor(
                [self.gt_model['b_Y']]).unsqueeze(1).to(torch.float))
            self.register_buffer('gt_beta', torch.from_numpy(
                self.gt_model['beta']).to(torch.float))
            self.register_buffer('gt_alpha_I', torch.from_numpy(
                self.gt_model['alpha_I']).unsqueeze(1).to(torch.float))
            self.register_buffer('gt_alpha_A', torch.from_numpy(
                self.gt_model['alpha_A']).unsqueeze(1).to(torch.float))

    def init_weights(self):
        # let's be extra careful not to oversaturate sigmoid, vanishing gradients should not be a problem so
        # I choose a small standard deviation (such that, for an input ~ Gaussian(0, 1), the output ~ Gaussian(0, 0.01))
        nn.init.normal_(self.alpha_A, mean=0, std=0.01 /
                        math.sqrt(self.nr_table_feats))
        nn.init.normal_(self.alpha_I, mean=0, std=0.01 /
                        math.sqrt(self.nr_image_feats))
        nn.init.normal_(self.beta, mean=0, std=0.01 /
                        math.sqrt(self.nr_table_feats))
        nn.init.constant_(self.b_Y, val=0)
        # we choose the other parameters such that f_img ~ Gaussian(0, 1), so initialize sigma_I to 1
        nn.init.constant_(self.log_var_I, val=0)

        if self.use_aux_as_output:
            nn.init.normal_(self.phi, mean=0, std=0.01 /
                            math.sqrt(self.nr_image_feats))
            # we standardize the J, so initializing sigma_J to 1 makes sense
            nn.init.constant_(self.log_var_J, val=0)

        # such that: f_img ~ Gaussian(0, 1)
        nn.init.normal_(self.reduce_im.weight, mean=0,
                        std=math.sqrt(self.nr_image_feats / 2048))

    def nll_batch(self, images, cont_feats, cat_feats, aux_cont_feats, targets, modes):
        # calculate the negative log-likelihood for one batch
        # also return predictions for convenience

        # check via shape of `images` whether we need to run through CNN or not
        if torch.numel(images[0]) == self.nr_image_feats:
            img_features = images
        else:
            # only forward valid images through CNN
            has_image = (modes == 0) | (modes == 1) | (
                modes == 2) | (modes == 4)
            img_features = torch.zeros(targets.size(
                0), self.nr_image_feats, dtype=torch.float, device=self.device)
            if has_image.any():
                cnn_out = self.cnn(images[has_image])

                img_features[has_image] = self.reduce_im(
                    cnn_out).view(cnn_out.size(0), -1)

        if self.use_tab and self.use_aux_as_input:
            missing_aux = aux_cont_feats.isnan().any(dim=1)
            cont_feats = torch.cat((cont_feats, aux_cont_feats), dim=1)
            cat_feats = cat_feats
        elif self.use_tab:
            missing_aux = [False for _ in aux_cont_feats]
            cont_feats = cont_feats
            cat_feats = cat_feats
        elif self.use_aux_as_input:
            missing_aux = aux_cont_feats.isnan().any(dim=1)
            cont_feats = aux_cont_feats
            cat_feats = [c.new_tensor(data=[]) for c in cat_feats]
            modes = [0 for _ in modes]

        loss_batch = 0
        mu_y_batch = []
        # during inference, `aux_cont_feats` will just be all NaNs
        for img, cont, cat, aux_cont, target, m, a in zip(img_features, cont_feats, cat_feats, aux_cont_feats, targets, modes, missing_aux):

            # add dim for calculations in forward
            img = img.unsqueeze(1)
            cont = cont.unsqueeze(1)
            cat = cat.unsqueeze(1)
            aux_cont = aux_cont.unsqueeze(1)
            target = target.unsqueeze(1)

            if m == 0:
                if a:
                    loss, mu_y = self.forward_missing_cont(
                        img, cont, cat, aux_cont, target)
                else:
                    loss, mu_y = self.forward_complete(
                        img, cont, cat, aux_cont, target)
            elif m == 1:
                loss, mu_y = self.forward_missing_cont(
                    img, cont, cat, aux_cont, target)
            elif m == 2:
                if a:
                    loss, mu_y = self.forward_missing_cont_cat(
                        img, cont, cat, aux_cont, target)
                else:
                    loss, mu_y = self.forward_missing_cat(
                        img, cont, cat, aux_cont, target)
            elif m == 3:
                if a:
                    loss, mu_y = self.forward_missing_cont_img(
                        cont, cat, aux_cont, target)
                else:
                    loss, mu_y = self.forward_missing_img(
                        cont, cat, aux_cont, target)
            elif m == 4:
                loss, mu_y = self.forward_missing_cont_cat(
                    img, cont, cat, aux_cont, target)
            elif m == 5:
                loss, mu_y = self.forward_missing_cont_img(
                    cont, cat, aux_cont, target)
            elif m == 6:
                if a:
                    loss, mu_y = self.forward_missing_cont_cat_img(
                        cont, cat, aux_cont, target)
                else:
                    loss, mu_y = self.forward_missing_cat_img(
                        cont, cat, aux_cont, target)
            elif m == 7:
                loss, mu_y = self.forward_missing_cont_cat_img(
                    cont, cat, aux_cont, target)

            loss_batch += loss
            mu_y_batch.append(mu_y)

        loss_batch /= targets.size(0)

        return loss_batch, torch.cat(mu_y_batch)

    def forward_complete(self, f_img, cont_table, cat_table, cont_aux, target):
        A = torch.cat((cont_table, cat_table), dim=0)
        mu_I = self.beta.t() @ A
        cov_I = torch.diag(
            torch.exp(self.log_var_I).repeat(self.nr_image_feats))
        logits_y = self.alpha_I.t() @ (f_img - mu_I) + self.alpha_A.t() @ A + self.b_Y

        loss = self.boltzmann_nll(logits_y, target, torch.exp(self.log_var_y))
        loss += self.gaussian_nll(f_img, mu_I, cov_I)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if valid_J_mask.any() and self.use_aux_as_output:
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
        mu_A[mu_A != mu_A] = 0.
        cov_A = all_feats.isnan().float().squeeze(1)
        cov_A = torch.diag(cov_A)

        mu_I = self.beta.t() @ mu_A
        cov_I = torch.diag(torch.exp(self.log_var_I).repeat(
            self.nr_image_feats)) + self.beta.t() @ cov_A @ self.beta

        num = torch.exp(self.log_var_I) * self.alpha_I.t() + \
            self.alpha_A.t() @ cov_A @ self.beta
        inv_denom = torch.inverse(cov_I)

        mu_p = self.alpha_A.t() @ mu_A + num @ inv_denom @ (f_img - mu_I) + self.b_Y
        var_p = self.alpha_A.t() @ cov_A @ self.alpha_A + \
            torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I - \
            num @ inv_denom @ num.t()

        logits_y = mu_p / (1. + math.pi / 8. * var_p) ** 0.5

        loss = self.boltzmann_nll(logits_y, target, torch.exp(self.log_var_y))
        loss += self.gaussian_nll(f_img, mu_I, cov_I)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if valid_J_mask.any() and self.use_aux_as_output:
            valid_phi = self.phi[:, valid_J_mask]
            mu_J = valid_phi.t() @ f_img
            cov_J = torch.diag(torch.exp(self.log_var_J[valid_J_mask]))
            loss += self.gaussian_nll(cont_aux[valid_J_mask], mu_J, cov_J)

        return loss, torch.sigmoid(logits_y).detach()

    def forward_missing_cat(self, f_img, cont_table, cat_table, cont_aux, target):
        # just remember: whenever possible, try to stay in log space!
        nan_idx = cat_table.isnan().nonzero(as_tuple=True)
        fill_range = torch.tensor(
            [-1, 1], dtype=cat_table.dtype, device=self.device)
        # more mem, but faster than moving elements of itertools.product to device?
        fill_values = torch.stack(
            [g.flatten() for g in torch.meshgrid([fill_range] * len(nan_idx[0]))], dim=1)

        cov_I_v = torch.diag(
            torch.exp(self.log_var_I).repeat(self.nr_image_feats))
        mu_y = 0
        ll = []
        for val in fill_values:
            cat_table_v = torch.index_put(cat_table, nan_idx, val)
            ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
            A_v = torch.cat((cont_table, cat_table_v), dim=0)
            mu_I_v = self.beta.t() @ A_v
            logits_y_v = self.alpha_I.t() @ (f_img - mu_I_v) + \
                self.alpha_A.t() @ A_v + self.b_Y
            ll_y_v = -self.boltzmann_nll(logits_y_v,
                                         target, torch.exp(self.log_var_y))
            ll_I_v = -self.gaussian_nll(f_img, mu_I_v, cov_I_v)
            ll.append(ll_A_v + ll_y_v + ll_I_v)

            mu_y += (torch.sigmoid(logits_y_v) * torch.exp(ll_A_v)).detach()

        loss = -torch.logsumexp(torch.stack(ll, dim=0), dim=0)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if valid_J_mask.any() and self.use_aux_as_output:
            valid_phi = self.phi[:, valid_J_mask]
            mu_J = valid_phi.t() @ f_img
            cov_J = torch.diag(torch.exp(self.log_var_J[valid_J_mask]))
            loss += self.gaussian_nll(cont_aux[valid_J_mask], mu_J, cov_J)

        return loss, mu_y

    def forward_missing_img(self, cont_table, cat_table, cont_aux, target):
        A = torch.cat((cont_table, cat_table), dim=0)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if (not valid_J_mask.any()) or not self.use_aux_as_output:
            logits_y = self.alpha_A.t() @ A + self.b_Y
            loss = self.boltzmann_nll(
                logits_y, target, torch.exp(self.log_var_y))

        else:
            valid_phi = self.phi[:, valid_J_mask]
            mu_J = valid_phi.t() @ self.beta.t() @ A
            cov_J = torch.diag(torch.exp(
                self.log_var_J[valid_J_mask])) + torch.exp(self.log_var_I) * valid_phi.t() @ valid_phi
            num = torch.exp(self.log_var_I) * self.alpha_I.t() @ valid_phi
            inv_denom = torch.inverse(cov_J)
            mu_p = self.alpha_A.t() @ A + \
                num @ inv_denom @ (cont_aux[valid_J_mask] - mu_J) + self.b_Y
            var_p = torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I - \
                num @ inv_denom @ num.t()

            logits_y = mu_p / (1. + math.pi / 8. * var_p) ** 0.5
            loss = self.boltzmann_nll(
                logits_y, target, torch.exp(self.log_var_y))
            loss += self.gaussian_nll(cont_aux[valid_J_mask], mu_J, cov_J)

        return loss, torch.sigmoid(logits_y).detach()

    def forward_missing_cont_cat(self, f_img, cont_table, cat_table, cont_aux, target):
        nan_idx = cat_table.isnan().nonzero(as_tuple=True)
        fill_range = torch.tensor(
            [-1, 1], dtype=cat_table.dtype, device=self.device)
        # more mem, but faster than moving elements of itertools.product to device?
        fill_values = torch.stack(
            [g.flatten() for g in torch.meshgrid([fill_range] * len(nan_idx[0]))], dim=1)

        mu_y = 0
        ll = []
        for val in fill_values:
            cat_table_v = torch.index_put(cat_table, nan_idx, val)
            ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
            A_v = torch.cat((cont_table, cat_table_v), dim=0)

            mu_A_v = A_v.clone()
            mu_A_v[mu_A_v != mu_A_v] = 0.
            cov_A_v = A_v.isnan().float().squeeze(1)
            cov_A_v = torch.diag(cov_A_v)

            mu_I_v = self.beta.t() @ mu_A_v
            cov_I_v = torch.diag(torch.exp(self.log_var_I).repeat(
                self.nr_image_feats)) + self.beta.t() @ cov_A_v @ self.beta

            num = torch.exp(self.log_var_I) * self.alpha_I.t() + \
                self.alpha_A.t() @ cov_A_v @ self.beta
            inv_denom = torch.inverse(cov_I_v)

            mu_p_v = self.alpha_A.t() @ mu_A_v + num @ inv_denom @ (f_img - mu_I_v) + self.b_Y
            var_p_v = self.alpha_A.t() @ cov_A_v @ self.alpha_A + \
                torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I - \
                num @ inv_denom @ num.t()

            logits_y_v = mu_p_v / (1. + math.pi / 8. * var_p_v) ** 0.5

            ll_y_v = -self.boltzmann_nll(logits_y_v,
                                         target, torch.exp(self.log_var_y))
            ll_I_v = -self.gaussian_nll(f_img, mu_I_v, cov_I_v)
            ll.append(ll_A_v + ll_y_v + ll_I_v)

            mu_y += (torch.sigmoid(logits_y_v) * torch.exp(ll_A_v)).detach()

        loss = -torch.logsumexp(torch.stack(ll, dim=0), dim=0)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if valid_J_mask.any() and self.use_aux_as_output:
            valid_phi = self.phi[:, valid_J_mask]
            mu_J = valid_phi.t() @ f_img
            cov_J = torch.diag(torch.exp(self.log_var_J[valid_J_mask]))
            loss += self.gaussian_nll(cont_aux[valid_J_mask], mu_J, cov_J)

        return loss, mu_y

    def forward_missing_cont_img(self, cont_table, cat_table, cont_aux, target):

        all_feats = torch.cat((cont_table, cat_table), dim=0)
        mu_A = all_feats.clone()
        mu_A[mu_A != mu_A] = 0.
        cov_A = all_feats.isnan().float().squeeze(1)
        cov_A = torch.diag(cov_A)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)
        if valid_J_mask.any() and self.use_aux_as_output:

            valid_phi = self.phi[:, valid_J_mask]
            mu_J = valid_phi.t() @ self.beta.t() @ mu_A
            cov_J = valid_phi.t() @ (torch.diag(torch.exp(self.log_var_I).repeat(self.nr_image_feats)) + self.beta.t() @ cov_A @ self.beta) @ valid_phi + \
                torch.diag(torch.exp(self.log_var_J[valid_J_mask]))
            num = (torch.exp(self.log_var_I) * self.alpha_I.t() +
                   self.alpha_A.t() @ cov_A @ self.beta) @ valid_phi
            inv_denom = torch.inverse(cov_J)

            mu_p = self.alpha_A.t() @ mu_A + \
                num @ inv_denom @ (cont_aux[valid_J_mask] - mu_J) + self.b_Y
            var_p = self.alpha_A.t() @ cov_A @ self.alpha_A + \
                torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I - \
                num @ inv_denom @ num.t()

            logits_y = mu_p / (1. + math.pi / 8. * var_p) ** 0.5

            loss = self.boltzmann_nll(
                logits_y, target, torch.exp(self.log_var_y))
            loss += self.gaussian_nll(cont_aux[valid_J_mask], mu_J, cov_J)

        else:
            mu_p = self.alpha_A.t() @ mu_A + self.b_Y
            var_p = self.alpha_A.t() @ cov_A @ self.alpha_A

            logits_y = mu_p / (1. + math.pi / 8. * var_p) ** 0.5
            loss = self.boltzmann_nll(
                logits_y, target, torch.exp(self.log_var_y))

        return loss, torch.sigmoid(logits_y).detach()

    def forward_missing_cat_img(self, cont_table, cat_table, cont_aux, target):
        nan_idx = cat_table.isnan().nonzero(as_tuple=True)
        fill_range = torch.tensor(
            [-1, 1], dtype=cat_table.dtype, device=self.device)
        # more mem, but faster than moving elements of itertools.product to device?
        fill_values = torch.stack(
            [g.flatten() for g in torch.meshgrid([fill_range] * len(nan_idx[0]))], dim=1)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)

        mu_y = 0
        ll = []
        if (not valid_J_mask.any()) or not self.use_aux_as_output:
            for val in fill_values:
                cat_table_v = torch.index_put(cat_table, nan_idx, val)
                ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
                A_v = torch.cat((cont_table, cat_table_v), dim=0)
                logits_y_v = self.alpha_A.t() @ A_v + self.b_Y

                ll_y_v = - \
                    self.boltzmann_nll(logits_y_v, target,
                                       torch.exp(self.log_var_y))
                ll.append(ll_A_v + ll_y_v)

                mu_y += (torch.sigmoid(logits_y_v) *
                         torch.exp(ll_A_v)).detach()

        else:
            valid_phi = self.phi[:, valid_J_mask]
            for val in fill_values:
                cat_table_v = torch.index_put(cat_table, nan_idx, val)
                ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
                A_v = torch.cat((cont_table, cat_table_v), dim=0)
                mu_J_v = valid_phi.t() @ self.beta.t() @ A_v
                cov_J_v = torch.diag(torch.exp(
                    self.log_var_J[valid_J_mask])) + torch.exp(self.log_var_I) * valid_phi.t() @ valid_phi
                num = torch.exp(self.log_var_I) * self.alpha_I.t() @ valid_phi
                inv_denom = torch.inverse(cov_J_v)
                mu_p_v = self.alpha_A.t() @ A_v + \
                    num @ inv_denom @ (cont_aux[valid_J_mask] -
                                       mu_J_v) + self.b_Y
                var_p_v = torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I - \
                    num @ inv_denom @ num.t()

                logits_y_v = mu_p_v / (1. + math.pi / 8. * var_p_v) ** 0.5

                ll_y_v = - \
                    self.boltzmann_nll(logits_y_v, target,
                                       torch.exp(self.log_var_y))
                ll_J_v = - \
                    self.gaussian_nll(cont_aux[valid_J_mask], mu_J_v, cov_J_v)
                ll.append(ll_A_v + ll_y_v + ll_J_v)

                mu_y += (torch.sigmoid(logits_y_v) *
                         torch.exp(ll_A_v)).detach()

        loss = -torch.logsumexp(torch.stack(ll, dim=0), dim=0)

        return loss, mu_y

    def forward_missing_cont_cat_img(self, cont_table, cat_table, cont_aux, target):
        nan_idx = cat_table.isnan().nonzero(as_tuple=True)
        fill_range = torch.tensor(
            [-1, 1], dtype=cat_table.dtype, device=self.device)
        # more mem, but faster than moving elements of itertools.product to device?
        fill_values = torch.stack(
            [g.flatten() for g in torch.meshgrid([fill_range] * len(nan_idx[0]))], dim=1)

        valid_J_mask = ~cont_aux.isnan().squeeze(1)

        mu_y = 0
        ll = []
        if (not valid_J_mask.any()) or not self.use_aux_as_output:
            for val in fill_values:
                cat_table_v = torch.index_put(cat_table, nan_idx, val)
                ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
                A_v = torch.cat((cont_table, cat_table_v), dim=0)

                mu_A_v = A_v.clone()
                mu_A_v[mu_A_v != mu_A_v] = 0.
                cov_A_v = A_v.isnan().float().squeeze(1)
                cov_A_v = torch.diag(cov_A_v)

                mu_p_v = self.alpha_A.t() @ mu_A_v + self.b_Y
                var_p_v = self.alpha_A.t() @ cov_A_v @ self.alpha_A

                logits_y_v = mu_p_v / (1. + math.pi / 8. * var_p_v) ** 0.5

                ll_y_v = - \
                    self.boltzmann_nll(logits_y_v, target,
                                       torch.exp(self.log_var_y))
                ll.append(ll_A_v + ll_y_v)

                mu_y += (torch.sigmoid(logits_y_v) *
                         torch.exp(ll_A_v)).detach()

        else:
            valid_phi = self.phi[:, valid_J_mask]
            for val in fill_values:
                cat_table_v = torch.index_put(cat_table, nan_idx, val)
                ll_A_v = self.get_cat_joint_logprob(val, nan_idx[0])
                A_v = torch.cat((cont_table, cat_table_v), dim=0)

                mu_A_v = A_v.clone()
                mu_A_v[mu_A_v != mu_A_v] = 0.
                cov_A_v = A_v.isnan().float().squeeze(1)
                cov_A_v = torch.diag(cov_A_v)

                mu_J_v = valid_phi.t() @ self.beta.t() @ mu_A_v
                cov_J_v = valid_phi.t() @ (torch.diag(torch.exp(self.log_var_I).repeat(self.nr_image_feats)) + self.beta.t() @ cov_A_v @ self.beta) @ valid_phi + \
                    torch.diag(torch.exp(self.log_var_J[valid_J_mask]))

                num = (torch.exp(self.log_var_I) * self.alpha_I.t() +
                       self.alpha_A.t() @ cov_A_v @ self.beta) @ valid_phi
                inv_denom = torch.inverse(cov_J_v)

                mu_p_v = self.alpha_A.t() @ mu_A_v + \
                    num @ inv_denom @ (cont_aux[valid_J_mask] -
                                       mu_J_v) + self.b_Y
                var_p_v = self.alpha_A.t() @ cov_A_v @ self.alpha_A + \
                    torch.exp(self.log_var_I) * self.alpha_I.t() @ self.alpha_I - \
                    num @ inv_denom @ num.t()

                logits_y_v = mu_p_v / (1. + math.pi / 8. * var_p_v) ** 0.5

                ll_y_v = - \
                    self.boltzmann_nll(logits_y_v, target,
                                       torch.exp(self.log_var_y))
                ll_J_v = - \
                    self.gaussian_nll(cont_aux[valid_J_mask], mu_J_v, cov_J_v)
                ll.append(ll_A_v + ll_y_v + ll_J_v)

                mu_y += (torch.sigmoid(logits_y_v) *
                         torch.exp(ll_A_v)).detach()

        loss = -torch.logsumexp(torch.stack(ll, dim=0), dim=0)

        return loss, mu_y

    def get_cat_joint_logprob(self, sampled_vals, indices):
        # we have a Bernoulli distribution
        ks = (sampled_vals + 1.) / 2.  # to range {0, 1}
        ps = self.cat_table_prior[indices]
        log_probs = ks * torch.log(ps) + (1 - ks) * \
            torch.log(1 - ps)  # Bernoulli
        return torch.sum(log_probs)

    def boltzmann_nll(self, logit, target, temperature):
        # negative log-likelihood
        return nn.functional.binary_cross_entropy_with_logits(logit / temperature, target, pos_weight=self.pos_weight, reduction='none')

    def gaussian_nll(self, value, mu, cov_matrix):
        # negative log-likelihood
        sigma_matrix = torch.cholesky(cov_matrix)
        diff = (value - mu).squeeze(1)
        M = torch.distributions.multivariate_normal._batch_mahalanobis(
            sigma_matrix, diff)
        half_log_det = sigma_matrix.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        # the constant term (no gradient) is included for now
        return 0.5 * (mu.shape[0] * math.log(2 * math.pi) + M) + half_log_det

    def training_step(self, batch, batch_idx):
        images = batch['image']
        cont_table = batch['cont_table']
        cat_table = batch['cat_table']
        cont_aux = batch['cont_imtable']
        modes = batch['modes']
        targets = batch['label']

        loss, _ = self.nll_batch(
            images, cont_table, cat_table, cont_aux, targets, modes)

        self.log('train_loss', loss)
        self.log('alpha_A_norm', torch.norm(self.alpha_A))
        self.log('alpha_I_norm', torch.norm(self.alpha_I))
        self.log('beta_norm', torch.norm(self.beta))
        self.log('b_Y_norm', torch.norm(self.b_Y))
        self.log('sigma_I', torch.sqrt(torch.exp(self.log_var_I)))

        if self.use_aux_as_output:
            self.log('phi_norm', torch.norm(self.phi))
            self.log('sigma_J', torch.mean(
                torch.sqrt(torch.exp(self.log_var_J))))

        return loss

    def on_validation_epoch_start(self):
        for m in self.valid_metrics_list:
            m.reset()

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        cont_table = batch['cont_table']
        cat_table = batch['cat_table']
        cont_aux = batch['cont_imtable']
        modes = batch['modes']
        targets = batch['label']

        loss, preds = self.nll_batch(
            images, cont_table, cat_table, cont_aux, targets, modes)
        self.log('valid_loss', loss)
        for m in self.valid_metrics_list:
            # Calling forward here since using wandb.watch() adds hooks which trigger error. Like this, all hooks are bypassed.
            m.forward(preds, targets.long())

        return preds

    def validation_epoch_end(self, outputs):
        # create a histogram with predictions
        all_predictions = torch.cat(outputs, dim=0).squeeze(1).tolist()
        # need to put commit=False here to prevent: https://github.com/PyTorchLightning/pytorch-lightning/pull/5050
        self.logger.experiment.log(
            {'valid_predictions': wandb.Histogram(all_predictions)}, commit=False)

        for m in self.valid_metrics_list:
            res = m.compute()
            self.log('valid_' + m.__class__.__name__, res)
            m.reset()

        if self.gt_model is not None:

            # evaluation for synthetic data
            alpha_A = self.alpha_A.squeeze(1).detach().cpu().numpy()
            self.log('abs_error_alpha_A', np.linalg.norm(
                self.gt_model['alpha_A'] - alpha_A))
            self.log('rel_error_alpha_A', np.linalg.norm(
                self.gt_model['alpha_A'] - alpha_A) / np.linalg.norm(self.gt_model['alpha_A']))
            fig_handle = plt.figure()
            plt.plot(alpha_A.flatten(), marker='.',
                     linestyle='None', label='estimate')
            plt.plot(self.gt_model['alpha_A'].flatten(),
                     marker='.', linestyle='None', label='target')
            plt.xlabel('element index')
            plt.ylabel('value')
            plt.grid(True)
            plt.legend()
            plt.title('alpha_A values, step {}'.format(self.global_step))
            self.logger.experiment.log(
                {'alpha_A_values': wandb.Image(fig_handle)}, commit=False)
            plt.close()

            alpha_I = self.alpha_I.squeeze(1).detach().cpu().numpy()
            self.log('abs_error_alpha_I', np.linalg.norm(
                self.gt_model['alpha_I'] - alpha_I))
            self.log('rel_error_alpha_I', np.linalg.norm(
                self.gt_model['alpha_I'] - alpha_I) / np.linalg.norm(self.gt_model['alpha_I']))
            fig_handle = plt.figure()
            plt.plot(alpha_I.flatten(), marker='.',
                     linestyle='None', label='estimate')
            plt.plot(self.gt_model['alpha_I'].flatten(),
                     marker='.', linestyle='None', label='target')
            plt.xlabel('element index')
            plt.ylabel('value')
            plt.grid(True)
            plt.legend()
            plt.title('alpha_I values, step {}'.format(self.global_step))
            self.logger.experiment.log(
                {'alpha_I_values': wandb.Image(fig_handle)}, commit=False)
            plt.close()

            beta = self.beta.detach().cpu().numpy()
            self.log('abs_error_beta', np.linalg.norm(
                self.gt_model['beta'] - beta))
            self.log('rel_error_beta', np.linalg.norm(
                self.gt_model['beta'] - beta) / np.linalg.norm(self.gt_model['beta']))
            fig_handle = plt.figure()
            plt.plot(beta.flatten(), marker='.',
                     linestyle='None', label='estimate')
            plt.plot(self.gt_model['beta'].flatten(),
                     marker='.', linestyle='None', label='target')
            plt.xlabel('element index')
            plt.ylabel('value')
            plt.grid(True)
            plt.legend()
            plt.title('beta values, step {}'.format(self.global_step))
            self.logger.experiment.log(
                {'beta_values': wandb.Image(fig_handle)}, commit=False)
            plt.close()

            if self.use_aux_as_output:
                phi = self.phi.detach().cpu().numpy()
                self.log('abs_error_phi', np.linalg.norm(
                    self.gt_model['phi'] - phi))
                self.log('rel_error_phi', np.linalg.norm(
                    self.gt_model['phi'] - phi) / np.linalg.norm(self.gt_model['phi']))
                fig_handle = plt.figure()
                plt.plot(phi.flatten(), marker='.',
                         linestyle='None', label='estimate')
                plt.plot(self.gt_model['phi'].flatten(
                ), marker='.', linestyle='None', label='target')
                plt.xlabel('element index')
                plt.ylabel('value')
                plt.grid(True)
                plt.legend()
                plt.title('phi values, step {}'.format(self.global_step))
                self.logger.experiment.log(
                    {'phi_values': wandb.Image(fig_handle)}, commit=False)
                plt.close()

            b_Y = self.b_Y.flatten().detach().cpu().numpy()
            self.log('abs_error_b_Y', np.linalg.norm(
                self.gt_model['b_Y'] - b_Y))
            self.log('rel_error_b_Y', np.linalg.norm(
                self.gt_model['b_Y'] - b_Y) / np.linalg.norm(self.gt_model['b_Y']))

            sigma_I = torch.sqrt(torch.exp(self.log_var_I)
                                 ).detach().cpu().numpy()
            self.log('abs_error_sigma_I', np.linalg.norm(
                self.gt_model['sigma_I'] - sigma_I))
            self.log('rel_error_sigma_I', np.linalg.norm(
                self.gt_model['sigma_I'] - sigma_I) / np.linalg.norm(self.gt_model['sigma_I']))

            if self.use_aux_as_output:
                sigma_J = torch.sqrt(
                    torch.exp(self.log_var_J)).detach().cpu().numpy()
                self.log('abs_error_sigma_J', np.linalg.norm(
                    self.gt_model['sigma_J'] - sigma_J))
                self.log('rel_error_sigma_J', np.linalg.norm(
                    self.gt_model['sigma_J'] - sigma_J) / np.linalg.norm(self.gt_model['sigma_J']))
                fig_handle = plt.figure()
                plt.plot(sigma_J.flatten(), marker='.',
                         linestyle='None', label='estimate')
                plt.plot(self.gt_model['sigma_J'].flatten(
                ), marker='.', linestyle='None', label='target')
                plt.xlabel('element index')
                plt.ylabel('value')
                plt.grid(True)
                plt.legend()
                plt.title('sigma_J values, step {}'.format(self.global_step))
                self.logger.experiment.log(
                    {'sigma_J_values': wandb.Image(fig_handle)}, commit=False)
                plt.close()

            fig_handle = plt.figure()
            plt.plot([b_Y, sigma_I], marker='.',
                     linestyle='None', label='estimate')
            plt.plot([self.gt_model['b_Y'], self.gt_model['sigma_I']],
                     marker='.', linestyle='None', label='target')
            plt.ylabel('value')
            plt.xticks([0, 1], ('b_Y', 'sigma_I'))
            plt.grid(True)
            plt.legend()
            plt.title('misc values, step {}'.format(self.global_step))
            self.logger.experiment.log(
                {'misc_values': wandb.Image(fig_handle)}, commit=False)
            plt.close()

    def on_test_epoch_start(self):
        for m in self.test_metrics_list:
            m.reset()

    def test_step(self, batch, batch_idx):
        images = batch['image']
        cont_table = batch['cont_table']
        cat_table = batch['cat_table']
        cont_aux = batch['cont_imtable']
        modes = batch['modes']
        targets = batch['label']

        loss, preds = self.nll_batch(
            images, cont_table, cat_table, cont_aux, targets, modes)
        self.log('test_loss', loss)
        for m in self.test_metrics_list:
            # Calling forward here since using wandb.watch() adds hooks which trigger error. Like this, all hooks are bypassed.
            m.forward(preds, targets.long())

        return preds

    def test_epoch_end(self, outputs):
        # create a histogram with predictions
        all_predictions = torch.cat(outputs, dim=0).squeeze(1).tolist()
        # need to put commit=False here to prevent: https://github.com/PyTorchLightning/pytorch-lightning/pull/5050
        self.logger.experiment.log(
            {'test_predictions': wandb.Histogram(all_predictions)}, commit=False)

        for m in self.test_metrics_list:
            res = m.compute()
            self.log('test_' + m.__class__.__name__, res)
            m.reset()

        if self.gt_model is not None:
            # evaluation for synthetic data
            alpha_A = self.alpha_A.squeeze(1).detach().cpu().numpy()
            self.log('abs_error_alpha_A', np.linalg.norm(
                self.gt_model['alpha_A'] - alpha_A))
            self.log('rel_error_alpha_A', np.linalg.norm(
                self.gt_model['alpha_A'] - alpha_A) / np.linalg.norm(self.gt_model['alpha_A']))
            fig_handle = plt.figure()
            plt.plot(alpha_A.flatten(), marker='.',
                     linestyle='None', label='estimate')
            plt.plot(self.gt_model['alpha_A'].flatten(),
                     marker='.', linestyle='None', label='target')
            plt.xlabel('element index')
            plt.ylabel('value')
            plt.grid(True)
            plt.legend()
            plt.title('alpha_A values, step {}'.format(self.global_step))
            self.logger.experiment.log(
                {'alpha_A_values': wandb.Image(fig_handle)}, commit=False)
            plt.close()

            alpha_I = self.alpha_I.squeeze(1).detach().cpu().numpy()
            self.log('abs_error_alpha_I', np.linalg.norm(
                self.gt_model['alpha_I'] - alpha_I))
            self.log('rel_error_alpha_I', np.linalg.norm(
                self.gt_model['alpha_I'] - alpha_I) / np.linalg.norm(self.gt_model['alpha_I']))
            fig_handle = plt.figure()
            plt.plot(alpha_I.flatten(), marker='.',
                     linestyle='None', label='estimate')
            plt.plot(self.gt_model['alpha_I'].flatten(),
                     marker='.', linestyle='None', label='target')
            plt.xlabel('element index')
            plt.ylabel('value')
            plt.grid(True)
            plt.legend()
            plt.title('alpha_I values, step {}'.format(self.global_step))
            self.logger.experiment.log(
                {'alpha_I_values': wandb.Image(fig_handle)}, commit=False)
            plt.close()

            beta = self.beta.detach().cpu().numpy()
            self.log('abs_error_beta', np.linalg.norm(
                self.gt_model['beta'] - beta))
            self.log('rel_error_beta', np.linalg.norm(
                self.gt_model['beta'] - beta) / np.linalg.norm(self.gt_model['beta']))
            fig_handle = plt.figure()
            plt.plot(beta.flatten(), marker='.',
                     linestyle='None', label='estimate')
            plt.plot(self.gt_model['beta'].flatten(),
                     marker='.', linestyle='None', label='target')
            plt.xlabel('element index')
            plt.ylabel('value')
            plt.grid(True)
            plt.legend()
            plt.title('beta values, step {}'.format(self.global_step))
            self.logger.experiment.log(
                {'beta_values': wandb.Image(fig_handle)}, commit=False)
            plt.close()

            if self.use_aux_as_output:
                phi = self.phi.detach().cpu().numpy()
                self.log('abs_error_phi', np.linalg.norm(
                    self.gt_model['phi'] - phi))
                self.log('rel_error_phi', np.linalg.norm(
                    self.gt_model['phi'] - phi) / np.linalg.norm(self.gt_model['phi']))
                fig_handle = plt.figure()
                plt.plot(phi.flatten(), marker='.',
                         linestyle='None', label='estimate')
                plt.plot(self.gt_model['phi'].flatten(
                ), marker='.', linestyle='None', label='target')
                plt.xlabel('element index')
                plt.ylabel('value')
                plt.grid(True)
                plt.legend()
                plt.title('phi values, step {}'.format(self.global_step))
                self.logger.experiment.log(
                    {'phi_values': wandb.Image(fig_handle)}, commit=False)
                plt.close()

            b_Y = self.b_Y.flatten().detach().cpu().numpy()
            self.log('abs_error_b_Y', np.linalg.norm(
                self.gt_model['b_Y'] - b_Y))
            self.log('rel_error_b_Y', np.linalg.norm(
                self.gt_model['b_Y'] - b_Y) / np.linalg.norm(self.gt_model['b_Y']))

            sigma_I = torch.sqrt(torch.exp(self.log_var_I)
                                 ).detach().cpu().numpy()
            self.log('abs_error_sigma_I', np.linalg.norm(
                self.gt_model['sigma_I'] - sigma_I))
            self.log('rel_error_sigma_I', np.linalg.norm(
                self.gt_model['sigma_I'] - sigma_I) / np.linalg.norm(self.gt_model['sigma_I']))

            if self.use_aux_as_output:
                sigma_J = torch.sqrt(
                    torch.exp(self.log_var_J)).detach().cpu().numpy()
                self.log('abs_error_sigma_J', np.linalg.norm(
                    self.gt_model['sigma_J'] - sigma_J))
                self.log('rel_error_sigma_J', np.linalg.norm(
                    self.gt_model['sigma_J'] - sigma_J) / np.linalg.norm(self.gt_model['sigma_J']))
                fig_handle = plt.figure()
                plt.plot(sigma_J.flatten(), marker='.',
                         linestyle='None', label='estimate')
                plt.plot(self.gt_model['sigma_J'].flatten(
                ), marker='.', linestyle='None', label='target')
                plt.xlabel('element index')
                plt.ylabel('value')
                plt.grid(True)
                plt.legend()
                plt.title('sigma_J values, step {}'.format(self.global_step))
                self.logger.experiment.log(
                    {'sigma_J_values': wandb.Image(fig_handle)}, commit=False)
                plt.close()

            fig_handle = plt.figure()
            plt.plot([b_Y, sigma_I], marker='.',
                     linestyle='None', label='estimate')
            plt.plot([self.gt_model['b_Y'], self.gt_model['sigma_I']],
                     marker='.', linestyle='None', label='target')
            plt.ylabel('value')
            plt.xticks([0, 1], ('b_Y', 'sigma_I'))
            plt.grid(True)
            plt.legend()
            plt.title('misc values, step {}'.format(self.global_step))
            self.logger.experiment.log(
                {'misc_values': wandb.Image(fig_handle)}, commit=False)
            plt.close()

    def train(self, mode=True):
        super().train(mode=mode)
        # set training flags properly
        if self.global_step < self.warmup_steps and mode:
            # entire CNN is frozen at this stage
            self.cnn.train(False)
        elif self.global_step >= self.warmup_steps and mode:
            self.cnn.train(self.fine_tune_cnn)
            # extra treatment for BN layers
            for m in self.cnn.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.train(not self.freeze_cnn_bn)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        """Use `on_train_batch_start` to thaw layers progressively."""
        if self.global_step == 0:
            utils_model.set_freeze(self.cnn, freeze_bn=True, freeze_other=True)

        if self.global_step == self.warmup_steps:
            utils_model.set_freeze(
                self.cnn, freeze_bn=self.freeze_cnn_bn, freeze_other=not self.fine_tune_cnn)

    def classifier_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith('cnn.'):
                yield param

    def cnn_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith('cnn.'):
                yield param

    def configure_optimizers(self):
        param_groups = [{'name': 'classifier',
                         'params': filter(lambda p: p.requires_grad, self.classifier_parameters()),
                         'lr': self.lr,
                         'weight_decay': self.weight_decay}]
        if self.fine_tune_cnn:
            param_groups.append({'name': 'cnn',
                                 'params': filter(lambda p: p.requires_grad, self.cnn_parameters()),
                                 'lr': self.lr / 10.,
                                 'weight_decay': self.weight_decay})
        optimizer = torch.optim.Adam(params=param_groups)
        if self.use_lr_scheduler:
            scheduler = {
                'scheduler': utils_model.WarmedUpInverseSquareRootLR(optimizer,
                                                                     warmup_epochs=self.warmup_steps),
                'interval': 'epoch'
            }
        else:
            # dummy
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1),
                'interval': 'step'
            }
        return [optimizer], [scheduler]
