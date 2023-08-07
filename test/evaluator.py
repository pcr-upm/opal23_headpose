import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class Evaluator:
    def __init__(self, ann_matrices, pred_matrices):
        assert ann_matrices.shape == pred_matrices.shape, "Shape mismatch between annotations and predictions: " \
                                                          f"{ann_matrices.shape} and {pred_matrices.shape}"
        self.ann_matrices = ann_matrices
        self.pred_matrices = pred_matrices

    def compute_mae(self):
        """
        Computes the Mean Absolute Error (MAE) amongst two batches of Euler angles.

        :returns: numpy.ndarray containing N MAE values between ground-truth and predicted Euler angles.
                  It computes the minimum between MAE(ann, pred) and MAE(ann, wrapped_pred) where wrapped_pred are
                  wrapped Euler angles. It also computes the 'Wrapped MAE' metric from Zhou et al. "WHENet: Real-time
                  Fine-Grained Estimation for Wide Range Head Pose":
                  min(MAE, 360 - MAE)
        """
        ann_angles = Rotation.from_matrix(self.ann_matrices).as_euler('XYZ', degrees=True)
        pred_angles = Rotation.from_matrix(self.pred_matrices).as_euler('XYZ', degrees=True)

        ann_angles[:, [0, 1]] = ann_angles[:, [1, 0]]  # PYR -> YPR
        pred_angles[:, [0, 1]] = pred_angles[:, [1, 0]]  # PYR -> YPR
        pred_wrap = self._wrap_angles(pred_angles)

        mae_ypr = np.abs(ann_angles - pred_angles)
        mae_ypr_wrap = np.abs(ann_angles - pred_wrap)

        diff = mae_ypr.mean(axis=-1)
        diff_wrap = mae_ypr_wrap.mean(axis=-1)

        mae_ypr[diff_wrap < diff] = mae_ypr_wrap[diff_wrap < diff]
        return np.minimum(mae_ypr, 360 - mae_ypr)

    def compute_ge(self, degrees=True):
        """
        Computes the geodesic error amongst ground-truth and predicted rotation matrices.

        :param degrees: True to return errors in degrees, False to return radians.
        :returns: numpy.ndarray containing N geodesic error values between ground-truth and predicted rotation matrices.
        """
        ann_pred_mult = np.matmul(self.ann_matrices, self.pred_matrices.transpose(0, 2, 1))

        error_radians = (np.trace(ann_pred_mult, axis1=1, axis2=2) - 1) / 2
        error_radians = np.clip(error_radians, -1, 1)
        error_radians = np.arccos(error_radians)

        if degrees:
            return np.rad2deg(error_radians)

        return error_radians

    def align_predictions(self, mask=None, tol=0.0001, max_iter=100000):
        """
        Aligns predicted rotation matrices to remove systematic errors entangled with network errors in cross-dataset
        evaluation.

        :param mask: iterable with the indices to use to compute the mean delta rotation. None: use all samples.
        :param tol: minimum error value needed to finish the optimization.
        :param max_iter: maximum number of iterations in the optimization loop.
        """
        if mask is None:
            mask = np.arange(self.ann_matrices.shape[0])

        deltas = np.matmul(self.pred_matrices[mask], self.ann_matrices[mask].transpose(0, 2, 1))
        mean_delta = self._compute_mean_rotation(deltas, tol, max_iter)
        self.pred_matrices[mask] = np.matmul(mean_delta.T, self.pred_matrices[mask])

    def _compute_mean_rotation(self, matrices, tol=0.0001, max_iter=100000):
        # Exclude samples outside the sphere of radius pi/2 for convergence
        distances = self._compute_displacement(np.eye(3), matrices)
        distances = np.linalg.norm(distances, axis=1)
        matrices = matrices[distances < np.pi/2]

        mean_matrix = matrices[0]
        for _ in range(max_iter):
            displacement = self._compute_displacement(mean_matrix, matrices)
            displacement = np.mean(displacement, axis=0)
            d_norm = np.linalg.norm(displacement)
            if d_norm < tol:
                break
            mean_matrix = mean_matrix @ cv2.Rodrigues(displacement)[0]

        return mean_matrix

    def _compute_displacement(self, mean_matrix, matrices):
        return np.concatenate([cv2.Rodrigues(r)[0].T for r in mean_matrix.T @ matrices])

    def _wrap_angles(self, angles):
        sign = np.sign(angles)
        wrapped_angles = np.array([180 - angles[:, 0],
                                   angles[:, 1] - sign[:, 1] * 180,
                                   angles[:, 2] - sign[:, 2] * 180])

        wrapped_angles[wrapped_angles > 180] -= 360
        wrapped_angles[wrapped_angles < -180] += 360
        return wrapped_angles.T
