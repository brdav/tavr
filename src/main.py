import gc

import jsonargparse
import pytorch_lightning as pl

import utils
from data.pl_data import DataModule
from probabilistic_model import ProbabilisticModel


def main(args):

    pl.seed_everything(args.datamodule.fold)

    # parse this in advance to adjust batch_size accordingly
    num_duplicates = utils.get_num_duplicates(
        args.trainer.gpus, args.trainer.num_nodes)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    data_version_callback = utils.RandomizeDataVersionCallback()
    exp_name = utils.get_experiment_name(args)

    # FIRST PHASE, USING EARLY STOP TO FIND ITERATION NUMBER
    dm = DataModule(num_duplicates=num_duplicates,
                    phase='tuning',
                    **vars(args.datamodule))
    dm.prepare_data()
    dm.setup('fit')
    model = ProbabilisticModel(metrics_list=dm.metrics_list,
                               nr_table_features=dm.nr_table_features,
                               nr_image_features=dm.nr_image_features,
                               nr_aux=dm.nr_aux,
                               gt_model=dm.gt_model,
                               pos_class_weight=dm.pos_class_weight,
                               cat_table_prior=dm.cat_table_prior,
                               **vars(args.model))

    logger = pl.loggers.WandbLogger(name='{}_tuning'.format(
        exp_name), save_dir=args.save_dir, project=args.project, config=utils.convert_namespace(args))

    early_stop_callback = utils.MyEarlyStopping(
        monitor='valid_AUROC', min_delta=1e-4, patience=10, mode='max')
    trainer = pl.Trainer(logger=logger,
                         callbacks=[lr_monitor, data_version_callback,
                                    early_stop_callback],
                         max_steps=model.iterations,
                         **vars(args.trainer))

    trainer.fit(model, datamodule=dm)

    optimal_step = model.global_step - early_stop_callback.patience * \
        len(dm.trainset) // dm.train_batch_size

    # make sure to clear up
    del dm
    logger.experiment.finish()
    del logger
    del model
    del trainer
    gc.collect()

    print('Minimum was reached at step {}'.format(optimal_step))

    # SECOND PHASE, RETRAINING
    dm = DataModule(num_duplicates=num_duplicates,
                    phase='final',
                    **vars(args.datamodule))
    dm.prepare_data()
    dm.setup('fit')
    model = ProbabilisticModel(metrics_list=dm.metrics_list,
                               nr_table_features=dm.nr_table_features,
                               nr_image_features=dm.nr_image_features,
                               nr_aux=dm.nr_aux,
                               gt_model=dm.gt_model,
                               pos_class_weight=dm.pos_class_weight,
                               cat_table_prior=dm.cat_table_prior,
                               **vars(args.model))
    logger = pl.loggers.WandbLogger(name='{}_final'.format(
        exp_name), save_dir=args.save_dir, project=args.project, config=utils.convert_namespace(args), reinit=True)
    trainer = pl.Trainer(logger=logger,
                         callbacks=[lr_monitor, data_version_callback],
                         max_steps=optimal_step,
                         **vars(args.trainer))

    trainer.fit(model, datamodule=dm)

    dm.setup('test')
    trainer.test(model, datamodule=dm)

    print('finished successfully')


if __name__ == '__main__':
    parser = utils.LightningArgumentParser()
    parser.add_argument(
        '--cfg', action=jsonargparse.ActionConfigFile, help='path to config file')
    parser.add_argument('--project', default='pytorch',
                        type=str, help='project name')
    parser.add_argument('--save_dir', default='./experiments',
                        type=str, help='experiment save dir')
    parser.add_argument('--misc', default='', type=str,
                        help='addition for exp_name')
    parser.add_argument('--use_tab', default=True, type=bool,
                        help='use tabular non-image features for prediction')
    parser.add_argument('--use_aux', default=False, type=bool,
                        help='use image features for prediction in logistic regression')
    parser.add_datamodule_args(DataModule)
    parser.add_model_args(ProbabilisticModel)
    parser.add_trainer_args()
    args = parser.parse_args()
    main(args)
