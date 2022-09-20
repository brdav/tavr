import io
import math

import matplotlib.pyplot as plt
import numpy as np
import requests
import SimpleITK as sitk
from scipy import interpolate
from src.data import utils_data


def extract_roi(image_path,
                landmark_path,
                size=(64, 64, 64),
                clip_range=(-200, 800),
                resolution=(0.8, 0.8, 0.8),
                augment=False,
                ):

    coord = np.load(landmark_path)
    assert len(coord) == 5, 'File {} does not have 5 landmarks.'.format(
        landmark_path)

    # spline parametrization
    params = [i / (size[2] - 1) for i in range(size[2])]

    # augment images by perturbing labels with noise
    noise_mean_dist = 5.
    # because we have a folded gaussian
    noise_std = noise_mean_dist * math.sqrt(math.pi / 2)

    sitk_image = sitk.ReadImage(image_path)
    sitk_image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    sitk_image = utils_data.clip_and_rescale_sitk_image(sitk_image, clip_range)
    sitk_image, size_factor = utils_data.resample_sitk_image(sitk_image,
                                                             resolution,
                                                             fill_value=0,
                                                             interpolator='linear',
                                                             return_factor=True)
    scaled_coord = np.round(size_factor * coord)

    if not augment:  # first dataset is ground truth
        noisy_coord = scaled_coord
    else:  # introduce noise
        rng = np.random.RandomState()
        noise_abs = noise_std * \
            rng.randn(*scaled_coord.shape) / \
            np.array(resolution)[np.newaxis, :]
        random_vecs = utils_data.random_three_vector(
            nr=scaled_coord.shape[0], random_state=rng)
        noisy_coord = scaled_coord + noise_abs * random_vecs

    tck, _ = interpolate.splprep(np.swapaxes(noisy_coord, 0, 1), k=3, s=100)

    # derivative is tangent to the curve
    points = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)
    Zs = np.swapaxes(interpolate.splev(params, tck, der=1), 0, 1)
    direc = np.array(sitk_image.GetDirection()[3:6])

    slices = []
    for i in range(len(Zs)):
        # I define the x'-vector as the projection of the y-vector onto the plane perpendicular to the spline
        xs = (direc - np.dot(direc, Zs[i]) /
              (np.power(np.linalg.norm(Zs[i]), 2)) * Zs[i])
        sitk_slice = utils_data.extract_slice_from_sitk_image(
            sitk_image, points[i], Zs[i], xs, list(size[:2]) + [1], fill_value=0)
        np_image = sitk.GetArrayFromImage(sitk_slice).transpose(2, 1, 0)
        # scale back to [0, 255]
        np_image = np.round(np_image * 255)
        slices.append(np_image)
    # stick slices together
    image_roi = np.concatenate(slices, axis=2)
    return image_roi


def main():
    """ This script assumes that a torso CT image and the
    anatomical landmark coordinates (as described in the
    paper) are given. For the automatic landmark localization,
    we used a two-stage pipeline, where the first step was based on
    https://github.com/amiralansary/rl-medical
    and the second step
    https://github.com/christianpayer/MedicalDataAugmentationTool-HeatmapRegression

    Purely for illustration purposes, we annotated a sample image 
    from the lung CT dataset of
    http://medicaldecathlon.com   --> lung_001.nii.gz

    Before executing this script download the sample:
    `wget https://data.vision.ee.ethz.ch/brdavid/tavi/sample_image.nii.gz`
    `wget https://data.vision.ee.ethz.ch/brdavid/tavi/sample_landmark.npy`
    """
    image_path = 'sample_image.nii.gz'
    landmark_path = 'sample_landmark.npy'
    # resulting roi
    roi = extract_roi(image_path, landmark_path)

    # assuming shape
    assert roi.shape == (64, 64, 64)
    fig, axs = plt.subplots(8, 8, figsize=(16, 16))

    for i in range(roi.shape[2]):
        h = i // 8
        w = i % 8
        axs[h, w].imshow(roi[:, :, i], cmap=plt.cm.gray, vmin=0, vmax=255)
        axs[h, w].axis('off')

    plt.show()


if __name__ == '__main__':
    main()
