import warnings

import torch
import torch.nn as nn


class WarmedUpInverseSquareRootLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Custom learning rate scheduler, combining learning rate warm up
    with a inverse square root schedule.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of iterations of warm up.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    """

    def __init__(self, optimizer, warmup_epochs, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.lr_fct = lambda epoch: 1.0 / (epoch + 1) ** 0.5
        self.warmup_epochs = warmup_epochs
        super(WarmedUpInverseSquareRootLR, self).__init__(
            optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will not be saved

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items(
        ) if key not in ('optimizer', 'lr_fct')}
        return state_dict

    def get_lr(self):

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        # warm up
        if self.last_epoch < self.warmup_epochs:
            lr_scale = min(1., float(self.last_epoch + 1) /
                           (self.warmup_epochs + 1))
            return [base_lr * lr_scale for base_lr in self.base_lrs]

        return [base_lr * self.lr_fct(self.last_epoch - self.warmup_epochs)
                for base_lr in self.base_lrs]


def _make_trainable(module: nn.Module, trainable: bool) -> None:
    # this should be the only place where we change `requires_grad` flag
    if trainable:
        for param in module.parameters():
            param.requires_grad = True
        module.train()
    else:
        for param in module.parameters():
            param.requires_grad = False
        module.eval()


def set_freeze(module: nn.Module,
               freeze_bn: bool = True,
               freeze_other: bool = True):
    _make_trainable(module, not freeze_other)
    # extra treatment for BN layers
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            _make_trainable(m, not freeze_bn)
