import math
import warnings

import numpy as np
import SimpleITK as sitk
import torch
from scipy import interpolate


TABULAR_NAMES = [
    "AVA",
    "Age",
    "Aortic_regurgitation",
    "BMI",
    "Creatinine",
    "Glomerular_filtration_rate",
    "Hemoglobin",
    "LVEF",
    "Mean_transaortic_pressure_gradient",
    "Mitral_regurgitation",
    "Cerebrovascular_disease",
    "Chronic_obstructive_pulmonary_disease",
    "Coronary_artery_bypass_grafting",
    "Coronary_atheromatosis_or_stenosis",
    "Diabetes_mellitus",
    "Dyslipidemia",
    "Family_history_of_any_cardiovascular_disease",
    "Male_sex",
    "Hypertension",
    "Pacemaker_at_baseline",
    "Peripheral_artery_disease",
    "Previous_cardiovascular_interventions",
    "Renal_replacement_or_dialysis",
    "Smoking_status",
    "Valve_in_valve",
]


MEASUREMENT_NAMES = [
    "Agatston_score_aortic_valve",
    "Area_derived_diameter_of_annulus_incl_calcification",
    "Area_of_annulus_incl_calcification",
    "Calcification_of_ascending_aorta",
    "Calcification_of_sinotubular_junction",
    "Diameter_of_ascending_aorta",
    "LVOT_area",
    "LVOT_maximal_diameter",
    "Maximal_annulus_diameter",
    "Maximal_diameter_of_sinotubular_junction",
    "Perimeter_of_annulus_incl_calcification",
    "Sinus_portion_maximal_diameter",
    "Volume_of_sinus_valsalva",
    "Volume_score_LVOT",
    "Volume_score_aortic_valve",
]


SITK_INTERPOLATOR_DICT = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "gaussian": sitk.sitkGaussian,
    "label_gaussian": sitk.sitkLabelGaussian,
    "bspline": sitk.sitkBSpline,
    "hamming_sinc": sitk.sitkHammingWindowedSinc,
    "cosine_windowed_sinc": sitk.sitkCosineWindowedSinc,
    "welch_windowed_sinc": sitk.sitkWelchWindowedSinc,
    "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc,
}


class InverseSquareRootLR(torch.optim.lr_scheduler._LRScheduler):
    """Inverse square root schedule."""

    def __init__(self, optimizer, warmup_epochs=0, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.lr_fct = lambda epoch: 1.0 / (epoch + 1) ** 0.5
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_fct")
        }
        return state_dict

    def get_lr(self):

        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        # warm up
        if self.last_epoch < self.warmup_epochs:
            lr_scale = min(1.0, float(self.last_epoch + 1) / (self.warmup_epochs + 1))
            return [base_lr * lr_scale for base_lr in self.base_lrs]

        return [
            base_lr * self.lr_fct(self.last_epoch - self.warmup_epochs)
            for base_lr in self.base_lrs
        ]


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
    assert np.dot(X, Z) < 1e-12, "the two input vectors are not perpendicular!"
    Y = np.cross(Z, X)

    orig_frame = np.array(orig_direction).reshape(num_dim, num_dim)
    new_frame = np.array([X, Y, Z])

    # important: when resampling images, the transform is used to map points from the output image space into the input image space
    rot_matrix = np.dot(orig_frame, np.linalg.pinv(new_frame))
    transform = sitk.AffineTransform(
        rot_matrix.flatten(), np.zeros(num_dim), rotation_center
    )

    phys_size = new_size * orig_spacing
    new_origin = rotation_center - phys_size / 2

    resampled_sitk_image = sitk.Resample(
        sitk_image,
        new_size,
        transform,
        sitk.sitkLinear,
        new_origin,
        orig_spacing,
        orig_direction,
        fill_value,
        orig_pixelid,
    )
    return resampled_sitk_image


def resample_sitk_image(
    sitk_image,
    new_spacing,
    interpolator=None,
    fill_value=0,
    anti_aliasing=True,
    return_factor=False,
):
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

    assert (
        interpolator in SITK_INTERPOLATOR_DICT.keys()
    ), "'interpolator' should be one of {}".format(SITK_INTERPOLATOR_DICT.keys())
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

    resampled_sitk_image = sitk.Resample(
        sitk_image,
        new_size,
        sitk.Transform(),
        sitk_interpolator,
        orig_origin,
        new_spacing,
        orig_direction,
        fill_value,
        orig_pixelid,
    )
    if return_factor:
        return resampled_sitk_image, size_factor
    return resampled_sitk_image


def align_sitk_image(sitk_image):
    if sitk_image.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        return sitk_image
    orig_direction = sitk_image.GetDirection()
    arr_direction = np.array(orig_direction).reshape(3, 3)
    assert np.all(arr_direction == np.diag(np.diag(arr_direction)))  # must be diagonal
    assert all(el in [-1.0, 1] for el in np.diag(arr_direction))
    flip_x = True if arr_direction[0, 0] == -1.0 else False
    flip_y = True if arr_direction[1, 1] == -1.0 else False
    flip_z = True if arr_direction[2, 2] == -1.0 else False
    flipped_sitk_image = sitk.Flip(sitk_image, [flip_x, flip_y, flip_z])
    assert flipped_sitk_image.GetDirection() == (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    return flipped_sitk_image


def clip_and_rescale_sitk_image(sitk_image, clip_range, data_type="float32"):
    """
    Clips a SimpleITK at a given intensity range and rescales the intensity scale to [0, 1]
    :param sitk_image: SimpleITK image
    :param clip_range: tuple, intensity interval to clip at
    :param data_type: data type of output image, for float32 image will be in range [0, 1],
                      for uint8 image will be in range [0, 255]
    :return: SimpleITK image
    """
    assert data_type in ["float32", "uint8"], "data_type must be float32 or uint8"
    data_type_dict = {"float32": sitk.sitkFloat32, "uint8": sitk.sitkUInt8}
    range_dict = {"float32": [0, 1], "uint8": [0, 255]}

    clipped_image = sitk.Clamp(sitk_image, data_type_dict[data_type], *clip_range)
    rescaled_image = sitk.RescaleIntensity(clipped_image, *range_dict[data_type])

    return rescaled_image


def crop_cube(image, center, size, scale=1):
    screen = np.zeros(size, dtype=np.float32)
    screen_xmin, screen_ymin, screen_zmin = 0, 0, 0
    screen_xmax, screen_ymax, screen_zmax = size
    # extract boundary locations
    xmin = center[0] - size[0] // 2 * scale
    ymin = center[1] - size[1] // 2 * scale
    zmin = center[2] - size[2] // 2 * scale
    xmax = center[0] + (size[0] + 1) // 2 * scale
    ymax = center[1] + (size[1] + 1) // 2 * scale
    zmax = center[2] + (size[2] + 1) // 2 * scale
    image_dims = image.shape
    # check if they violate image boundary and fix it
    if xmin < 0:
        xmin = 0
        screen_xmin = screen_xmax - len(range(0, xmax, scale))
    if ymin < 0:
        ymin = 0
        screen_ymin = screen_ymax - len(range(0, ymax, scale))
    if zmin < 0:
        zmin = 0
        screen_zmin = screen_zmax - len(range(0, zmax, scale))
    if xmax > image_dims[0]:
        xmax = image_dims[0]
        screen_xmax = screen_xmin + len(range(xmin, xmax, scale))
    if ymax > image_dims[1]:
        ymax = image_dims[1]
        screen_ymax = screen_ymin + len(range(ymin, ymax, scale))
    if zmax > image_dims[2]:
        zmax = image_dims[2]
        screen_zmax = screen_zmin + len(range(zmin, zmax, scale))
    screen[
        screen_xmin:screen_xmax, screen_ymin:screen_ymax, screen_zmin:screen_zmax
    ] = image[xmin:xmax:scale, ymin:ymax:scale, zmin:zmax:scale]
    return screen


def extract_roi(
    sitk_image,
    coords,
    size=(64, 64, 64),
    clip_range=(-200, 800),
    resolution=(0.8, 0.8, 0.8),
    augment=False,
):
    assert len(coords) == 5, "need 5 landmarks"
    assert sitk_image.GetDirection() == (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ), "image is not aligned, use `align_sitk_image` first"
    sitk_image = clip_and_rescale_sitk_image(sitk_image, clip_range)
    sitk_image, size_factor = resample_sitk_image(
        sitk_image, resolution, fill_value=0, interpolator="linear", return_factor=True
    )
    scaled_coords = np.round(size_factor * coords)

    if not augment:  # first dataset is ground truth
        noisy_coords = scaled_coords
    else:  # introduce noise
        # augment images by perturbing labels with noise
        noise_mean_dist = 5.0
        # because we have a folded gaussian
        noise_std = noise_mean_dist * math.sqrt(math.pi / 2)
        rng = np.random.RandomState()
        noise_abs = (
            noise_std
            * rng.randn(*scaled_coords.shape)
            / np.array(resolution)[np.newaxis, :]
        )
        random_vecs = random_three_vector(nr=scaled_coords.shape[0], random_state=rng)
        noisy_coords = scaled_coords + noise_abs * random_vecs

    tck, _ = interpolate.splprep(np.swapaxes(noisy_coords, 0, 1), k=3, s=100)

    # spline parametrization
    params = [i / (size[2] - 1) for i in range(size[2])]
    # derivative is tangent to the curve
    points = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)
    Zs = np.swapaxes(interpolate.splev(params, tck, der=1), 0, 1)
    direc = np.array(sitk_image.GetDirection()[3:6])

    slices = []
    for i in range(len(Zs)):
        # I define the x'-vector as the projection of the y-vector onto the plane perpendicular to the spline
        xs = direc - np.dot(direc, Zs[i]) / (np.power(np.linalg.norm(Zs[i]), 2)) * Zs[i]
        sitk_slice = extract_slice_from_sitk_image(
            sitk_image, points[i], Zs[i], xs, list(size[:2]) + [1], fill_value=0
        )
        np_image = sitk.GetArrayFromImage(sitk_slice).transpose(2, 1, 0)
        # scale back to [0, 255]
        np_image = np.round(np_image * 255)
        slices.append(np_image)
    # stick slices together
    image_roi = np.concatenate(slices, axis=2)
    return image_roi
