import cv2
import numpy as np


def align_predictions(ann, pred, tol=0.0001):
    """
    Aligns predictions to remove systematic errors entangled with prediction errors in cross-dataset evaluations.

    :param ann: N x 3 x 3 numpy.ndarray containing ground-truth rotation matrices.
    :param pred: N x 3 x 3 numpy.ndarray containing predicted rotation matrices.
    :param tol: minimum error value needed to finish the optimization.
    :returns: N x 3 x 3 numpy.ndarray containing the aligned rotation matrices.
    """

    def _average_displacement(avg_rot, ri):
        return np.mean(np.concatenate([cv2.Rodrigues(r)[0].T for r in avg_rot.T @ ri]), axis=0)

    deltas = np.matmul(pred.transpose(0, 2, 1), ann)

    # Exclude samples outside the sphere of radius pi/2 for convergence
    identity = np.expand_dims(np.eye(3), 0)
    identity = np.repeat(identity, deltas.shape[0], axis=0)
    gd = np.deg2rad(compute_ge(deltas, identity))
    deltas = deltas[gd < np.pi / 2]

    res = deltas[0, :, :]
    while True:
        displacement = _average_displacement(res, deltas)
        d_norm = np.linalg.norm(displacement)
        if d_norm < tol:
            break
        res = res @ cv2.Rodrigues(displacement)[0]

    return np.matmul(pred, res)


def compute_ge(m1, m2):
    """
    Computes the geodesic error amongst two batches of rotation matrices.

    :param m1: N x 3 x 3 numpy.ndarray containing the first batch of rotation matrices.
    :param m2: N x 3 x 3 numpy.ndarray containing the second batch of rotation matrices.
    :returns: numpy.ndarray containing N geodesic error values between m1 and m2.
    """
    m = np.matmul(m1, m2.transpose(0, 2, 1))

    cos = (np.trace(m, axis1=1, axis2=2) - 1) / 2
    cos = np.clip(cos, -1, 1)

    return np.rad2deg(np.arccos(cos))


def compute_mae(anns, pred, use_pyr_format=False):
    """
    Computes the Mean Absolute Error (MAE) amongst two batches of Euler angles.

    :param anns: N x 3 numpy.ndarray containing the first batch of Euler angles.
    :param pred: N x 3 numpy.ndarray containing the second batch of Euler angles.
    :param use_pyr_format: flag to indicate the order of rotation. True means pitch-yaw-roll order, False means
                           yaw-pitch-roll order.
    :returns: numpy.ndarray containing N MAE values between anns and pred.
    """
    pred_wrap = _wrap_angles(pred, use_pyr_format)

    mae_ypr = np.abs(anns - pred)
    mae_ypr_wrap = np.abs(anns - pred_wrap)

    diff = mae_ypr.mean(axis=-1)
    diff_wrap = mae_ypr_wrap.mean(axis=-1)

    mae_ypr[diff_wrap < diff, :] = mae_ypr_wrap[diff_wrap < diff, :]
    return np.minimum(mae_ypr, 360 - mae_ypr)


def _wrap_angles(euler, use_pyr_format=False):
    sign = np.sign(euler)
    if use_pyr_format:
        wrapped_euler = np.array([180 - euler[:, 0], euler[:, 1] - sign[:, 1] * 180, euler[:, 2] - sign[:, 2] * 180])
    else:
        wrapped_euler = np.hstcak([euler[:, 0] - sign[:, 0] * 180, 180 - euler[:, 1], euler[:, 2] - sign[:, 2] * 180])

    wrapped_euler[wrapped_euler > 180] -= 360
    wrapped_euler[wrapped_euler < -180] += 360
    return wrapped_euler.T


def convert_rotation(pose, to, use_pyr_format=False):
    """
    Converts any pose representation to another. Currently supports Euler, quaternion, 6D and rotation matrix
    representations.

    :param pose: numpy.ndarray containing a rotation defined in Euler, quaternion, 6D or rotation matrix representation.
    :param to: string that indicates the target representation. Can be 'euler', 'quaternion', 'ortho6d' or 'matrix'.
    :param use_pyr_format: flag to indicate the order of rotation in Euler angles. True means pitch-yaw-roll order,
                           False means yaw-pitch-roll order.
    :returns: numpy.ndarray containing the converted pose array.
    """
    pose = np.array(pose)

    if to == 'euler':
        if pose.size == 4:
            pose = _quaternion_to_rotation_matrix(pose)
        if pose.size == 6:
            pose = _ortho6d_to_rotation_matrix(pose)
        if pose.size == 9:
            pose = pose.reshape(3, 3)
            return _rotation_matrix_to_euler_pyr(pose) if use_pyr_format else _rotation_matrix_to_euler(pose)
    elif to == 'quaternion':
        if pose.size == 3:
            pose = _euler_to_rotation_matrix_pyr(pose) if use_pyr_format else _euler_to_rotation_matrix(pose)
        if pose.size == 6:
            pose = _ortho6d_to_rotation_matrix(pose)
        if pose.size == 9:
            pose = pose.reshape(3, 3)
            return _rotation_matrix_to_quaternion(pose)
    elif to == 'ortho6d':
        if pose.size == 3:
            pose = _euler_to_rotation_matrix_pyr(pose) if use_pyr_format else _euler_to_rotation_matrix(pose)
        if pose.size == 4:
            pose = _quaternion_to_rotation_matrix(pose)
        if pose.size == 9:
            return pose.reshape(3, 3)[:, :2].T.flatten()
    elif to == 'matrix':
        if pose.size == 3:
            return _euler_to_rotation_matrix_pyr(pose) if use_pyr_format else _euler_to_rotation_matrix(pose)
        if pose.size == 4:
            return _quaternion_to_rotation_matrix(pose)
        if pose.size == 6:
            return _ortho6d_to_rotation_matrix(pose)
        if pose.size == 9:
            return pose.reshape(3, 3)
    else:
        raise ValueError(f"Argument 'to' has an invalid value: {to}. "
                         "Accepted values are 'euler', 'quaternion', 'ortho6d', 'matrix'")

    return pose


def _rotation_matrix_to_euler(rot_matrix):
    # http://euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
    a00, a01, a02 = rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2]
    a10, a11, a12 = rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2]
    a20, a21, a22 = rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2]
    if abs(1.0 - a10) <= np.finfo(float).eps:  # singularity at north pole / special case a10 == 1
        yaw   = np.arctan2(a02,a22)
        pitch = np.pi/2.0
        roll  = 0
    elif abs(-1.0 - a10) <= np.finfo(float).eps:  # singularity at south pole / special case a10 == -1
        yaw   = np.arctan2(a02,a22)
        pitch = -np.pi/2.0
        roll  = 0
    else:  # standard case
        yaw   = np.arctan2(-a20, a00)
        pitch = np.arcsin(a10)
        roll  = np.arctan2(-a12, a11)
    # Convert to degrees
    euler = np.rad2deg(np.array([yaw, pitch, roll]))
    # Change coordinates system
    euler = np.array([(-euler[0])+90, -euler[1], (-euler[2])-90])
    euler[euler > 180] -= 360
    euler[euler < -180] += 360
    return euler


def _rotation_matrix_to_euler_pyr(rot_matrix):
    a00, a01, a02 = rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2]
    a10, a11, a12 = rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2]
    a20, a21, a22 = rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2]

    if abs(1.0 - a20) <= np.finfo(float).eps:  # Gimbal lock: yaw = 90º
        yaw = np.pi / 2
        pitch = np.arctan2(a12, a11)
        roll = 0
    elif abs(-1.0 - a20) <= np.finfo(float).eps:  # Gimbal lock: yaw = -90º
        yaw = -np.pi / 2
        pitch = np.arctan2(a12, a11)
        roll = 0
    else:  # standard case
        yaw = np.arcsin(a20)
        cy = np.cos(yaw)
        pitch = -np.arctan2(a21 / cy, a22 / cy)
        roll = -np.arctan2(a10 / cy, a00 / cy)
        if abs(pitch) > np.pi / 2 and abs(roll) > np.pi / 2:
            yaw = np.pi - yaw
            cy = np.cos(yaw)
            pitch = -np.arctan2(a21 / cy, a22 / cy)
            roll = -np.arctan2(a10 / cy, a00 / cy)

    euler = np.rad2deg(np.array([yaw, pitch, roll]))
    euler[euler > 180] -= 360
    euler[euler < -180] += 360
    return euler


def _rotation_matrix_to_quaternion(rot_matrix):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    m00, m01, m02 = rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2]
    m10, m11, m12 = rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2]
    m20, m21, m22 = rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2]

    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) & (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif (m11 > m22):
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    quat = np.array([qw, qx, qy, qz])
    if quat[0] < 0:
        quat = -quat
    return quat


def _euler_to_rotation_matrix(headpose):
    # http://euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
    # Change coordinates system
    euler = np.array([-(headpose[0] - 90), -headpose[1], -(headpose[2] + 90)])
    # Convert to radians
    rad = np.deg2rad(euler)
    cy = np.cos(rad[0])
    sy = np.sin(rad[0])
    cp = np.cos(rad[1])
    sp = np.sin(rad[1])
    cr = np.cos(rad[2])
    sr = np.sin(rad[2])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])  # yaw
    Rp = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])  # pitch
    Rr = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])  # roll
    return np.matmul(np.matmul(Ry, Rp), Rr)


def _euler_to_rotation_matrix_pyr(angles):
    rad = np.deg2rad(angles)
    cy = np.cos(rad[0])
    sy = np.sin(rad[0])
    cp = np.cos(rad[1])
    sp = np.sin(rad[1])
    cr = np.cos(rad[2])
    sr = np.sin(rad[2])

    Rp = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]])
    Ry = np.array([[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]])
    Rr = np.array([[cr, sr, 0.0], [-sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return Rr.dot(Ry).dot(Rp)


def _ortho6d_to_rotation_matrix(poses):
    x_raw = poses[0:3]
    y_raw = poses[3:6]

    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)

    x = x.reshape(3, 1)
    y = y.reshape(3, 1)
    z = z.reshape(3, 1)
    matrix = np.concatenate((x, y, z), 1)
    return matrix


def _quaternion_to_rotation_matrix(quaternion):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    sqw = qw * qw
    sqx = qx * qx
    sqy = qy * qy
    sqz = qz * qz

    # invs (inverse square length) is only required if quaternion is not already normalized
    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = (sqx - sqy - sqz + sqw) * invs  # since sqw + sqx + sqy + sqz =1/invs*invs
    m11 = (-sqx + sqy - sqz + sqw) * invs
    m22 = (-sqx - sqy + sqz + sqw) * invs

    tmp1 = qx * qy
    tmp2 = qz * qw
    m10 = 2.0 * (tmp1 + tmp2) * invs
    m01 = 2.0 * (tmp1 - tmp2) * invs

    tmp1 = qx * qz
    tmp2 = qy * qw
    m20 = 2.0 * (tmp1 - tmp2) * invs
    m02 = 2.0 * (tmp1 + tmp2) * invs

    tmp1 = qy * qz
    tmp2 = qx * qw
    m21 = 2.0 * (tmp1 + tmp2) * invs
    m12 = 2.0 * (tmp1 - tmp2) * invs

    rot_matrix = np.array([[m00, m01, m02],
                           [m10, m11, m12],
                           [m20, m21, m22]])
    return rot_matrix
