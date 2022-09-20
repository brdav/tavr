import copy

import jsonargparse
import pytorch_lightning as pl


class RandomizeDataVersionCallback(pl.callbacks.Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        # this will prompt the dataset to randomly pick another version
        trainer.datamodule.trainset.image_dataset = None


class MyEarlyStopping(pl.callbacks.early_stopping.EarlyStopping):

    def on_validation_end(self, trainer, pl_module):
        # We train for at least 100 epochs
        if trainer.current_epoch >= max(pl_module.warmup_steps, 100):
            self._run_early_stopping_check(trainer)


class LightningArgumentParser(jsonargparse.ArgumentParser):
    """
    Extension of jsonargparse.ArgumentParser that lets us parse datamodule, model and training args.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_datamodule_args(self, datamodule_obj: pl.LightningDataModule):
        """Add arguments from datamodule_obj to the parser
        Args:
            datamodule_obj (pl.LightningDataModule): Any LightningDataModule subclass
        """
        # Ugly workaround, really hope they fix argparse in pytorch lightning soon
        # https://github.com/PyTorchLightning/pytorch-lightning-bolts/pull/293
        skip = {'num_duplicates', 'phase'}
        self.add_method_arguments(
            datamodule_obj, '__init__', 'datamodule', as_group=True, skip=skip)

    def add_model_args(self, model_obj: pl.LightningModule):
        """Add arguments from model_obj to the parser
        Args:
            model_obj (pl.LightningModule): Any LightningModule subclass
        """
        skip = {'metrics_list', 'cat_table_prior', 'nr_table_features',
                'nr_aux', 'pos_class_weight', 'gt_model', 'nr_image_features'}
        self.add_class_arguments(model_obj, 'model', as_group=True, skip=skip)

    def add_trainer_args(self, trainer_obj: pl.Trainer = pl.Trainer):
        """Add Lightning's Trainer args to the parser.
        Args:
            trainer_obj (pl.Trainer, optional): The trainer object to add arguments from. Defaults to pl.Trainer.
        """
        skip = {'logger', 'callbacks', 'plugins',
                'resume_from_checkpoint', 'max_steps', 'max_epochs'}
        self.add_class_arguments(
            pl.Trainer, 'trainer', as_group=True, skip=skip)


def get_num_duplicates(gpus, num_nodes):
    # parse `gpus` arg as in pl:
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/299de5dc6235fa22d57a0e1720788c3c40b85ecd/pytorch_lightning/accelerators/accelerator_connector.py#L103
    gpu_ids = pl.utilities.device_parser.parse_gpu_ids(gpus)
    num_gpus = 0 if gpu_ids is None else len(gpu_ids)
    return max(1, num_gpus * num_nodes)


def convert_namespace(namespace, do_copy=True):
    if do_copy:
        namespace = copy.deepcopy(namespace)
    if hasattr(namespace, '__dict__'):
        dict_namespace = vars(namespace)
        for k, v in dict_namespace.items():
            dict_namespace[k] = convert_namespace(v, do_copy=False)
        return dict_namespace
    return namespace


def get_experiment_name(args):
    cargs = copy.deepcopy(args)
    ll = []
    ll.append('ProbabilisticModel')
    ll.append('fold' + str(cargs.datamodule.fold))
    if len(cargs.misc) > 0:
        ll.append(cargs.misc)
    return '_'.join(ll)
