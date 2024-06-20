import argparse
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
import torch
from src.model import ProbabilisticModel


def predict(opt):
    """Predict on CPU using sample data."""

    print("Load the model...")
    if opt.model == "resnet":
        model = ProbabilisticModel.load_from_checkpoint(
            "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/679121/tavr_3d_resnet50_checkpoint.ckpt",
            map_location="cpu",
        )
    elif opt.model == "swin":
        model = ProbabilisticModel.load_from_checkpoint(
            "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/679121/tavr_swin_unetr_checkpoint.ckpt",
            map_location="cpu",
        )
    else:
        raise ValueError
    model.eval()

    print("Load the data...")
    sitk_image = sitk.ReadImage(opt.image) if opt.image else None
    tabular = pd.read_json(opt.tabular, typ="series") if opt.tabular else None
    measurements = (
        pd.read_json(opt.measurements, typ="series") if opt.measurements else None
    )

    print("Predict...")
    with torch.inference_mode():
        start = timer()
        prediction, roi = model(
            image=sitk_image, tabular=tabular, measurements=measurements
        )
        end = timer()
    print("TAVR risk score: ", prediction.item())
    print("Execution time: {} seconds".format(end - start))

    # visualize the ROI
    if roi is not None:
        assert roi.shape == (64, 64, 64)
        _, axs = plt.subplots(8, 8, figsize=(16, 16), num="ROI slices")
        for i in range(roi.shape[2]):
            h = i // 8
            w = i % 8
            axs[h, w].imshow(roi[:, :, i], cmap=plt.cm.gray, vmin=0, vmax=255)
            axs[h, w].axis("off")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="swin", help="choose between `resnet` and `swin`"
    )
    parser.add_argument(
        "--tabular",
        type=str,
        default=None,
        help="path to JSON file with tabular patient features",
    )
    parser.add_argument(
        "--measurements",
        type=str,
        default=None,
        help="path to JSON file with image measurements",
    )
    parser.add_argument("--image", type=str, default=None, help="path to image file")
    opt = parser.parse_args()
    predict(opt)
