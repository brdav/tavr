import warnings
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from . import transforms, utils_data
from .synthetic import SyntheticDataset


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 num_duplicates: int,
                 dataset_name: str,
                 data_dir: str = './',
                 raw_table_dir: str = None,
                 raw_nifti_dir: str = None,
                 fold: int = 0,
                 phase: str = 'final',
                 batch_size: Union[int, str] = 8,
                 num_workers: int = 4,
                 load_images: bool = True,
                 exclude_procedure_data: bool = False,
                 ignore_non_image: bool = False,
                 nr_image_features: int = 20,
                 missing_mode: str = None,
                 infinite_flag: bool = False,
                 do_imputation: bool = False,
                 imputation: str = 'mean',
                 impute_with_cont: bool = True,
                 impute_with_cat: bool = True,
                 impute_with_imcont: bool = True,
                 metrics_compute_on_step: bool = False,
                 auto_landmarks: bool = False,
                 load_extracted_features: bool = False,
                 eliminate_feature: str = None,
                 dataset_size: int = None,
                 missing_factor: float = None):
        super().__init__()

        if isinstance(batch_size, int):
            assert batch_size % num_duplicates == 0
        else:
            assert num_duplicates == 1
        self.batch_size = batch_size

        self.num_duplicates = num_duplicates
        self.data_dir = data_dir
        self.raw_table_dir = raw_table_dir
        self.raw_nifti_dir = raw_nifti_dir
        self.fold = fold
        self.phase = phase
        self.load_images = load_images
        self.exclude_procedure_data = exclude_procedure_data
        self.ignore_non_image = ignore_non_image
        self.num_workers = num_workers
        self.missing_mode = missing_mode
        self.infinite_flag = infinite_flag
        self.do_imputation = do_imputation
        self.imputation = imputation
        self.impute_with_cont = impute_with_cont
        self.impute_with_cat = impute_with_cat
        self.impute_with_imcont = impute_with_imcont
        self.nr_image_features = nr_image_features
        self.auto_landmarks = auto_landmarks
        self.load_extracted_features = load_extracted_features
        self.eliminate_feature = eliminate_feature
        self.dataset_size = dataset_size
        self.missing_factor = missing_factor

        self.dataset_cls = globals()[dataset_name]

        with warnings.catch_warnings():  # because of AUROC warning
            warnings.simplefilter("ignore")
            self.metrics_list = nn.ModuleList([
                torchmetrics.Accuracy(compute_on_step=metrics_compute_on_step),
                torchmetrics.Precision(
                    compute_on_step=metrics_compute_on_step),
                torchmetrics.Recall(compute_on_step=metrics_compute_on_step),
                torchmetrics.F1(
                    num_classes=1, compute_on_step=metrics_compute_on_step),
                torchmetrics.AUROC(
                    pos_label=1, compute_on_step=metrics_compute_on_step),
            ])

        self.train_transforms = transforms.Compose([
            transforms.RandomFlip() if self.load_images else lambda x: x,
            transforms.GaussianBlur() if self.load_images else lambda x: x,
            transforms.StandardizeImage() if self.load_images else lambda x: x
        ])
        self.test_transforms = transforms.Compose([
            transforms.StandardizeImage() if self.load_images else lambda x: x
        ])

    def prepare_data(self):
        # download
        self.dataset_cls(data_dir=self.data_dir,
                         raw_table_dir=self.raw_table_dir,
                         raw_nifti_dir=self.raw_nifti_dir,
                         split='train',
                         fold=self.fold,
                         load_images=self.load_images,
                         exclude_procedure_data=self.exclude_procedure_data,
                         ignore_non_image=self.ignore_non_image,
                         nr_image_features=self.nr_image_features,
                         missing_mode=self.missing_mode,
                         infinite_flag=self.infinite_flag,
                         do_imputation=self.do_imputation,
                         imputation=self.imputation,
                         impute_with_cont=self.impute_with_cont,
                         impute_with_cat=self.impute_with_cat,
                         impute_with_imcont=self.impute_with_imcont,
                         load_extracted_features=self.load_extracted_features,
                         eliminate_feature=self.eliminate_feature,
                         dataset_size=self.dataset_size,
                         missing_factor=self.missing_factor,
                         download=True)

    def setup(self, stage=None):

        # Assign train and valid dataset for use in dataloaders
        if stage == 'fit' or stage is None:

            if self.phase == 'tuning':
                self.trainset = self.dataset_cls(data_dir=self.data_dir,
                                                 raw_table_dir=self.raw_table_dir,
                                                 raw_nifti_dir=self.raw_nifti_dir,
                                                 split='train',
                                                 fold=self.fold,
                                                 load_images=self.load_images,
                                                 exclude_procedure_data=self.exclude_procedure_data,
                                                 ignore_non_image=self.ignore_non_image,
                                                 transforms=self.train_transforms,
                                                 nr_image_features=self.nr_image_features,
                                                 missing_mode=self.missing_mode,
                                                 infinite_flag=self.infinite_flag,
                                                 do_imputation=self.do_imputation,
                                                 imputation=self.imputation,
                                                 impute_with_cont=self.impute_with_cont,
                                                 impute_with_cat=self.impute_with_cat,
                                                 impute_with_imcont=self.impute_with_imcont,
                                                 load_extracted_features=self.load_extracted_features,
                                                 eliminate_feature=self.eliminate_feature,
                                                 dataset_size=self.dataset_size,
                                                 missing_factor=self.missing_factor)
                self.valset = self.dataset_cls(data_dir=self.data_dir,
                                               raw_table_dir=self.raw_table_dir,
                                               raw_nifti_dir=self.raw_nifti_dir,
                                               split='val',
                                               fold=self.fold,
                                               load_images=self.load_images,
                                               exclude_procedure_data=self.exclude_procedure_data,
                                               ignore_non_image=self.ignore_non_image,
                                               transforms=self.test_transforms,
                                               nr_image_features=self.nr_image_features,
                                               missing_mode=self.missing_mode,
                                               infinite_flag=self.infinite_flag,
                                               do_imputation=self.do_imputation,
                                               imputation=self.imputation,
                                               impute_with_cont=self.impute_with_cont,
                                               impute_with_cat=self.impute_with_cat,
                                               impute_with_imcont=self.impute_with_imcont,
                                               auto_landmarks=self.auto_landmarks,
                                               load_extracted_features=self.load_extracted_features,
                                               eliminate_feature=self.eliminate_feature,
                                               dataset_size=self.dataset_size,
                                               missing_factor=self.missing_factor)
                self.nr_table_features = self.trainset.nr_table_features
                self.nr_aux = self.trainset.nr_aux
                self.cat_table_prior = self.trainset.cat_table_prior
                self.pos_class_weight = self.trainset.pos_class_weight
                self.gt_model = self.trainset.gt_model if hasattr(
                    self.trainset, 'gt_model') else None
                if self.batch_size == 'full':
                    self.train_batch_size = len(self.trainset)
                    self.val_batch_size = len(self.valset)
                else:
                    self.train_batch_size = self.batch_size
                    self.val_batch_size = self.batch_size

            elif self.phase == 'final':
                self.trainset = self.dataset_cls(data_dir=self.data_dir,
                                                 raw_table_dir=self.raw_table_dir,
                                                 raw_nifti_dir=self.raw_nifti_dir,
                                                 split='trainval',
                                                 fold=self.fold,
                                                 load_images=self.load_images,
                                                 exclude_procedure_data=self.exclude_procedure_data,
                                                 ignore_non_image=self.ignore_non_image,
                                                 transforms=self.train_transforms,
                                                 nr_image_features=self.nr_image_features,
                                                 missing_mode=self.missing_mode,
                                                 infinite_flag=self.infinite_flag,
                                                 do_imputation=self.do_imputation,
                                                 imputation=self.imputation,
                                                 impute_with_cont=self.impute_with_cont,
                                                 impute_with_cat=self.impute_with_cat,
                                                 impute_with_imcont=self.impute_with_imcont,
                                                 load_extracted_features=self.load_extracted_features,
                                                 eliminate_feature=self.eliminate_feature,
                                                 dataset_size=self.dataset_size,
                                                 missing_factor=self.missing_factor)
                self.nr_table_features = self.trainset.nr_table_features
                self.nr_aux = self.trainset.nr_aux
                self.cat_table_prior = self.trainset.cat_table_prior
                self.pos_class_weight = self.trainset.pos_class_weight
                self.gt_model = self.trainset.gt_model if hasattr(
                    self.trainset, 'gt_model') else None
                if self.batch_size == 'full':
                    self.train_batch_size = len(self.trainset)
                else:
                    self.train_batch_size = self.batch_size

            else:
                raise NotImplementedError

        if stage == 'test' or stage is None:
            self.testset = self.dataset_cls(data_dir=self.data_dir,
                                            raw_table_dir=self.raw_table_dir,
                                            raw_nifti_dir=self.raw_nifti_dir,
                                            split='test',
                                            fold=self.fold,
                                            load_images=self.load_images,
                                            exclude_procedure_data=self.exclude_procedure_data,
                                            ignore_non_image=self.ignore_non_image,
                                            transforms=self.test_transforms,
                                            nr_image_features=self.nr_image_features,
                                            missing_mode=self.missing_mode,
                                            infinite_flag=self.infinite_flag,
                                            do_imputation=self.do_imputation,
                                            imputation=self.imputation,
                                            impute_with_cont=self.impute_with_cont,
                                            impute_with_cat=self.impute_with_cat,
                                            impute_with_imcont=self.impute_with_imcont,
                                            auto_landmarks=self.auto_landmarks,
                                            load_extracted_features=self.load_extracted_features,
                                            eliminate_feature=self.eliminate_feature,
                                            dataset_size=self.dataset_size,
                                            missing_factor=self.missing_factor)
            self.nr_table_features = self.testset.nr_table_features
            self.nr_aux = self.testset.nr_aux
            if self.batch_size == 'full':
                self.test_batch_size = len(self.testset)
            else:
                self.test_batch_size = self.batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset,
                                           shuffle=True,
                                           batch_size=self.train_batch_size // self.num_duplicates,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           drop_last=True,
                                           worker_init_fn=utils_data.worker_seed_init_fn_)

    def val_dataloader(self):
        if self.phase == 'tuning':
            return torch.utils.data.DataLoader(self.valset,
                                               batch_size=self.val_batch_size // self.num_duplicates,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               worker_init_fn=utils_data.worker_seed_init_fn_)
        else:
            return None

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset,
                                           batch_size=self.test_batch_size // self.num_duplicates,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           worker_init_fn=utils_data.worker_seed_init_fn_)
