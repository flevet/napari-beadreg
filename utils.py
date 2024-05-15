from PyQt5.QtWidgets import QMessageBox
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from enum import Enum
import json
import cv2 as cv
import numpy as np

def check_is_valid_int(n, err_msg, min_val=1, max_val=None):
    show_err = False
    if isinstance(n, float):
        if int(n) != n:
            show_err = True
    elif not isinstance(n, int):
        show_err = True

    if not show_err:
        show_err = n < min_val if min_val is not None else show_err
        show_err = show_err and (n > max_val if max_val is not None else show_err)

    if show_err:
        mess = QMessageBox()
        mess.setText('Invalid input: {}.'.format(err_msg))
        mess.exec()


def check_is_valid_float(value, err_msg, min_val=0., max_val=None):
    show_err = False
    try:
        value = float(value)
    except Exception as e:
        show_err = True
        err_msg += f"\n{repr(e)}"

    if not show_err:
        show_err = value < min_val if min_val is not None else show_err
        show_err = show_err and (value > max_val if max_val is not None else show_err)

    if show_err:
        mess = QMessageBox()
        mess.setText('Invalid input: {}.'.format(err_msg))
        mess.exec()

    return value

def is_palmtracer2_file(filepath: str):
    if filepath.endswith('.txt'):
        with open(filepath) as f:
            first_line = f.readline()
            if first_line.startswith('Width') and 'Spectral' in first_line:
                return True
    return False

class RegistrationBase:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def save_params(self, path):
        params = self.get_params_in_json_convertible_format()

        with open(path, "w") as f:
            json.dump(params, f)

    def get_params_in_json_convertible_format(self):
        pass

class ThinPlateSpineTransform(RegistrationBase):
    scaling_factor: float = 1.
    source_points: np.ndarray = None
    target_points: np.ndarray = None

    def __init__(self, X: np.ndarray = None, Y: np.ndarray = None):
        """
        X: targets points
        Y: source points
        """
        super().__init__(self)  # , coef_=coef_, intercept_=intercept_)

        self.target_points = X.astype(np.float32)
        self.source_points = Y.astype(np.float32)

        self.determine_scale_factor(self.source_points, self.target_points)

        source_points = self.scale_point_cloud(self.source_points)
        target_points = self.scale_point_cloud(self.target_points)
        source_points = source_points.reshape(-1, len(source_points), 2)
        target_points = target_points.reshape(-1, len(target_points), 2)

        matches = list()
        for i in range(0, len(source_points[0])):
            matches.append(cv.DMatch(i, i, 0))

        self.tps_points = cv.createThinPlateSplineShapeTransformer()
        self.tps_points.estimateTransformation(source_points, target_points, matches)

        self.tps_images = cv.createThinPlateSplineShapeTransformer()
        self.tps_images.estimateTransformation(target_points, source_points, matches)

    #This is necessary because the spline transform does not seem to work with coordinates bigger than 5,000 / 10,000
    #Thus scaling back to 5,000
    def determine_scale_factor(self, X: np.ndarray = None, Y: np.ndarray = None):
        x_t, y_t = np.transpose(X), np.transpose(Y)
        max_x, max_y = max(np.max(x_t[0]), np.max(y_t[0])), max(np.max(x_t[1]), np.max(y_t[1]))
        max_v = max(max_x, max_y)
        self.scaling_factor = 1. if max_v < 5000 else 5000. / max_v

    def scale_point_cloud(self, X: np.ndarray):
        return X * self.scaling_factor

    def scale_point_cloud_back(self, X: np.ndarray):
        return X / self.scaling_factor

    def transform_point_cloud(self, Y: np.ndarray):
        """
        Y: source positions
        """
        Y = self.scale_point_cloud(Y)
        Y = Y.reshape(-1,len(Y),2)
        f32_Y = np.zeros(Y.shape, dtype=np.float32)
        f32_Y[:] = Y[:]
        transform_cost, new_pts1 = self.tps_points.applyTransformation(f32_Y)
        return self.scale_point_cloud_back(new_pts1[0])

    def transform_image(self, w: int, h: int):
        step: int = int(min(w, h) / 10)
        im = np.zeros((h, w, 1)).astype((np.uint8))
        # draw parallel grids
        for y in range(0, im.shape[0], step):
            im[y, :, :] = 255
        for x in range(0, im.shape[1], step):
            im[:, x, :] = 255
        new_im = self.tps_images.warpImage(im)
        return new_im