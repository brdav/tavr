import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from sklearn import model_selection


def stratified_random_split(dataset, split_proportion):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:
    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # we only want the indices
    indices_1, indices_2 = model_selection.train_test_split(list(range(len(dataset))),
                                                            test_size=split_proportion,
                                                            random_state=0,
                                                            shuffle=True,
                                                            stratify=dataset.labels)
    return [torch.utils.data.Subset(dataset, indices_1),
            torch.utils.data.Subset(dataset, indices_2)]


def worker_seed_init_fn_(*args):
    # make sure that each subprocess has a different random state,
    # e.g. for np.random this is not the case by default:
    # https://github.com/pytorch/pytorch/issues/5059
    torch_seed = torch.initial_seed()
    np.random.seed(torch_seed % 2 ** 32)


def random_three_vector(nr, random_state=None):
    """
    Generate 'nr' random 3D unit vectors (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    if random_state is None:
        random_state = np.random.random.__self__
    vec = np.zeros((nr, 3))
    for l in range(nr):
        phi = random_state.uniform(0, np.pi * 2)
        costheta = random_state.uniform(-1, 1)

        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        vec[l] = (x, y, z)
    return vec


SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def extract_slice_from_sitk_image(sitk_image, point, Z, X, new_size, fill_value=0):
    """
    Extract oblique slice from SimpleITK image. Efficient, because it rotates the grid and
    only samples the desired slice.

    """
    num_dim = sitk_image.GetDimension()

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())

    # SimpleITK expects lists, not ndarrays
    new_size = [int(el) for el in new_size]
    point = [float(el) for el in point]

    rotation_center = sitk_image.TransformContinuousIndexToPhysicalPoint(point)

    X = X / np.linalg.norm(X)
    Z = Z / np.linalg.norm(Z)
    assert np.dot(X, Z) < 1e-12, 'the two input vectors are not perpendicular!'
    Y = np.cross(Z, X)

    orig_frame = np.array(orig_direction).reshape(num_dim, num_dim)
    new_frame = np.array([X, Y, Z])

    # important: when resampling images, the transform is used to map points from the output image space into the input image space
    rot_matrix = np.dot(orig_frame, np.linalg.pinv(new_frame))
    transform = sitk.AffineTransform(
        rot_matrix.flatten(), np.zeros(num_dim), rotation_center)

    phys_size = new_size * orig_spacing
    new_origin = rotation_center - phys_size / 2

    resampled_sitk_image = sitk.Resample(sitk_image,
                                         new_size,
                                         transform,
                                         sitk.sitkLinear,
                                         new_origin,
                                         orig_spacing,
                                         orig_direction,
                                         fill_value,
                                         orig_pixelid)
    return resampled_sitk_image


def resample_sitk_image(sitk_image,
                        new_spacing,
                        interpolator=None,
                        fill_value=0,
                        anti_aliasing=True,
                        return_factor=False):
    """
    Resamples an ITK image to a new grid.
    :param sitk_image: SimpleITK image
    :param new_spacing: tuple, specifying the output spacing
    :param interpolator: str, for example 'nearest' or 'linear'
    :param fill_value: int
    :param anti_aliasing: bool, whether to use smoothing before downsampling
    :param return_factor: bool
    :return: SimpleITK image
    """

    assert interpolator in SITK_INTERPOLATOR_DICT.keys(),\
        "'interpolator' should be one of {}".format(
            SITK_INTERPOLATOR_DICT.keys())
    sitk_interpolator = SITK_INTERPOLATOR_DICT[interpolator]

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = sitk_image.GetSpacing()
    orig_size = sitk_image.GetSize()

    size_factor = np.array(orig_spacing) / np.array(new_spacing)

    new_size = np.array(orig_size) * size_factor
    # image dimensions are in integers, SimpleITK expects lists
    new_size = [int(el) for el in new_size]
    new_spacing = [float(el) for el in new_spacing]

    if anti_aliasing:
        # just like in http://scikit-image.org/docs/dev/api/skimage.transform.html#resize
        sigma = (1 - size_factor) / 2
        # didn't find a prettier way to disable smoothing
        sigma[sigma <= 0] = 1e-12
        sitk_image = sitk.SmoothingRecursiveGaussian(sitk_image, sigma, False)

    resampled_sitk_image = sitk.Resample(sitk_image,
                                         new_size,
                                         sitk.Transform(),
                                         sitk_interpolator,
                                         orig_origin,
                                         new_spacing,
                                         orig_direction,
                                         fill_value,
                                         orig_pixelid)
    if return_factor:
        return resampled_sitk_image, size_factor
    return resampled_sitk_image


def clip_and_rescale_sitk_image(sitk_image, clip_range, data_type='float32'):
    """
    Clips a SimpleITK at a given intensity range and rescales the intensity scale to [0, 1]
    :param sitk_image: SimpleITK image
    :param clip_range: tuple, intensity interval to clip at
    :param data_type: data type of output image, for float32 image will be in range [0, 1],
                      for uint8 image will be in range [0, 255]
    :return: SimpleITK image
    """
    assert data_type in ['float32',
                         'uint8'], 'data_type must be float32 or uint8'
    data_type_dict = {'float32': sitk.sitkFloat32,
                      'uint8': sitk.sitkUInt8}
    range_dict = {'float32': [0, 1],
                  'uint8': [0, 255]}

    clipped_image = sitk.Clamp(
        sitk_image, data_type_dict[data_type], *clip_range)
    rescaled_image = sitk.RescaleIntensity(
        clipped_image, *range_dict[data_type])

    return rescaled_image
