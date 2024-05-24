import os
import urllib.request
from collections import Counter, deque
from typing import List

import numpy as np
import onnxruntime as ort
import SimpleITK as sitk
import torch

from .utils import clip_and_rescale_sitk_image, crop_cube, resample_sitk_image

model_urls = {
    "RLLandmarkLocalizer": "rl_localizer_model.onnx",
    "RegressionLandmarkLocalizer": "regression_localizer_model.onnx",
}


class CombinedLandmarkLocalizer:
    """Returns the landmark coordinates of the aligned sitk_image."""

    def __init__(self):
        super().__init__()
        self.localization_stage_1 = RLLandmarkLocalizer()
        self.localization_stage_2 = RegressionLandmarkLocalizer()

    def __call__(self, image):
        assert image.GetDirection() == (
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
        res_image = clip_and_rescale_sitk_image(image, (-200, 800), data_type="uint8")
        res_image, size_factor = resample_sitk_image(
            res_image,
            (2.0, 2.0, 2.0),
            fill_value=0,
            interpolator="linear",
            return_factor=True,
        )
        np_img = sitk.GetArrayFromImage(res_image).transpose(2, 1, 0)
        center = self.localization_stage_1(np_img)
        # confine the search space
        crop = crop_cube(np_img, center, (72, 72, 72)) / 255.0
        crop -= np.mean(crop)
        heatmap = self.localization_stage_2(crop)
        locs = np.array(
            [
                np.unravel_index(el.argmax(), el.shape)
                for el in heatmap.transpose(3, 0, 1, 2)
            ]
        )
        locs += center - np.array((36, 36, 36))
        return locs / size_factor


class RLLandmarkLocalizer:

    def __init__(
        self,
        max_num_frames: int = 1000,
        scale_space: List[int] = [8, 4, 2, 1],
        history_length: int = 20,
    ):
        self.max_num_frames = max_num_frames
        self.scale_space = scale_space
        self.history_length = history_length

        # the model was originally trained in TensorFlow, we therefore
        # provide it here in the ONNX format
        url = model_urls["RLLandmarkLocalizer"]
        filename = url.split("/")[-1]
        model_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            os.makedirs(model_dir, exist_ok=True)
            urllib.request.urlretrieve(url, cached_file)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 8
        self.ort_sess = ort.InferenceSession(
            cached_file, sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def __call__(self, image):
        state, location = self.new_game(image)

        while True:
            # 1. predict
            qvalues = self.ort_sess.run(
                None, {"images:0": state[np.newaxis, ..., np.newaxis]}
            )[0][0]
            action = np.argmax(qvalues)
            # 2. act
            location, state, terminate = self.step(action, qvalues, location)
            if terminate:
                break

        self.game_breakdown()
        return location

    def new_game(self, image, starting_point=(0.5, 0.5, 0.5)):
        """Set up a new game; so put agent at starting location and observe state."""
        self.image = image
        self.image_dims = image.shape

        self.cnt = 0  # counter to limit number of steps per episodes
        self._loc_history = [(0,) * 3] * self.history_length
        self._qvalues_history = [(0,) * 6] * self.history_length
        self.remaining_scales = deque(self.scale_space)
        self.current_scale = self.remaining_scales.popleft()

        location = tuple(int(p * d) for p, d in zip(starting_point, self.image_dims))
        state = crop_cube(self.image, location, (25, 25, 25), self.current_scale)
        return state, location

    def game_breakdown(self):
        del self.image
        del self.image_dims
        del self.cnt
        del self._loc_history
        del self._qvalues_history
        del self.remaining_scales
        del self.current_scale

    def step(self, act, qvalues, current_loc):
        """
        ---------------------------------------------------------------------------
        Copyright (c) 2017 tensorpack-medical. All rights reserved.

        This source code is licensed under the license found in the
        LICENSE file in https://github.com/amiralansary/rl-medical.
        ---------------------------------------------------------------------------
        """
        terminate = False
        # UP Z+ -----------------------------------------------------------
        if act == 0:
            next_location = (
                current_loc[0],
                current_loc[1],
                round(current_loc[2] + self.current_scale),
            )
            if next_location[2] >= self.image_dims[2]:
                next_location = current_loc

        # FORWARD Y+ ---------------------------------------------------------
        if act == 1:
            next_location = (
                current_loc[0],
                round(current_loc[1] + self.current_scale),
                current_loc[2],
            )
            if next_location[1] >= self.image_dims[1]:
                next_location = current_loc
        # RIGHT X+ -----------------------------------------------------------
        if act == 2:
            next_location = (
                round(current_loc[0] + self.current_scale),
                current_loc[1],
                current_loc[2],
            )
            if next_location[0] >= self.image_dims[0]:
                next_location = current_loc
        # LEFT X- -----------------------------------------------------------
        if act == 3:
            next_location = (
                round(current_loc[0] - self.current_scale),
                current_loc[1],
                current_loc[2],
            )
            if next_location[0] <= 0:
                next_location = current_loc
        # BACKWARD Y- ---------------------------------------------------------
        if act == 4:
            next_location = (
                current_loc[0],
                round(current_loc[1] - self.current_scale),
                current_loc[2],
            )
            if next_location[1] <= 0:
                next_location = current_loc
        # DOWN Z- -----------------------------------------------------------
        if act == 5:
            next_location = (
                current_loc[0],
                current_loc[1],
                round(current_loc[2] - self.current_scale),
            )
            if next_location[2] <= 0:
                next_location = current_loc
        # ---------------------------------------------------------------------

        self.cnt += 1
        self._update_history(next_location, qvalues)

        # terminate if maximum number of steps is reached
        if self.cnt >= self.max_num_frames:
            terminate = True

        # check if agent oscillates
        if self.oscillating:
            next_location = self.get_best_location()

            # multi-scale steps
            if len(self.remaining_scales) > 0:
                self.current_scale = self.remaining_scales.popleft()
                self._clear_history()
            # terminate if scale is 1
            else:
                terminate = True

        next_state = crop_cube(
            self.image, next_location, (25, 25, 25), self.current_scale
        )

        return next_location, next_state, terminate

    def get_best_location(self):
        """
        ---------------------------------------------------------------------------
        Copyright (c) 2017 tensorpack-medical. All rights reserved.

        This source code is licensed under the license found in the
        LICENSE file in https://github.com/amiralansary/rl-medical.
        ---------------------------------------------------------------------------

        Get best location with lowest max q-value from last four locations stored in history.
        """
        last_qvalues_history = self._qvalues_history[-4:]
        last_loc_history = self._loc_history[-4:]
        best_qvalues = np.max(last_qvalues_history, axis=1)
        best_idx = best_qvalues.argmin()
        best_location = last_loc_history[best_idx]
        return best_location

    def _clear_history(self):
        """
        ---------------------------------------------------------------------------
        Copyright (c) 2017 tensorpack-medical. All rights reserved.

        This source code is licensed under the license found in the
        LICENSE file in https://github.com/amiralansary/rl-medical.
        ---------------------------------------------------------------------------

        Clear history buffers.
        """
        self._loc_history = [(0,) * 3] * self.history_length
        self._qvalues_history = [(0,) * 6] * self.history_length

    def _update_history(self, location, qvalues):
        """
        ---------------------------------------------------------------------------
        Copyright (c) 2017 tensorpack-medical. All rights reserved.

        This source code is licensed under the license found in the
        LICENSE file in https://github.com/amiralansary/rl-medical.
        ---------------------------------------------------------------------------

        Update location history buffer and q-value history buffer.
        """
        # update location history
        self._loc_history[:-1] = self._loc_history[1:]
        self._loc_history[-1] = location
        # update q-value history
        self._qvalues_history[:-1] = self._qvalues_history[1:]
        self._qvalues_history[-1] = qvalues

    @property
    def oscillating(self):
        """
        ---------------------------------------------------------------------------
        Copyright (c) 2017 tensorpack-medical. All rights reserved.

        This source code is licensed under the license found in the
        LICENSE file in https://github.com/amiralansary/rl-medical.
        ---------------------------------------------------------------------------
        """
        counter = Counter(self._loc_history)
        freq = counter.most_common()
        if len(freq) == 1:  # to avoid index error
            return False
        if freq[0][0] == (0, 0, 0):  # if _loc_history hasn't filled up yet
            if freq[1][1] > 3:
                return True
            else:
                return False
        elif (
            freq[0][1] > 3
        ):  # if agent has been at same point more than 3 times it's oscillating
            return True
        return False


class RegressionLandmarkLocalizer:

    def __init__(self):
        # the model was originally trained in TensorFlow, we therefore
        # provide it here in the ONNX format
        url = model_urls["RegressionLandmarkLocalizer"]
        filename = url.split("/")[-1]
        model_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            os.makedirs(model_dir, exist_ok=True)
            urllib.request.urlretrieve(url, cached_file)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 8
        self.ort_sess = ort.InferenceSession(
            cached_file, sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def __call__(self, x):
        output = self.ort_sess.run(
            None,
            {
                "images:0": x[np.newaxis, ..., np.newaxis],
                "training_time:0": np.array(0, dtype=bool),
            },
        )[0]
        return np.squeeze(output, axis=0)
