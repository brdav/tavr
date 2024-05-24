from pytorch_lightning.cli import LightningCLI
from src.data_module import TAVRDataModule
from src.model import ProbabilisticModel


def cli_main():
    cli = LightningCLI(ProbabilisticModel,
                       TAVRDataModule,
                       save_config_kwargs={'overwrite': True})


if __name__ == '__main__':
    cli_main()
