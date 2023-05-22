import math

import torch
import torch.nn.functional as F


def convert_rotation(pose, to, use_pyr_format=False):
    def pose_size(pose):
        return torch.numel(pose) // pose.shape[0]

    if to == 'euler':
        if pose_size(pose) == 4:
            pose = _quaternion_to_rotation_matrix(pose)
        if pose_size(pose) == 6:
            pose = _ortho6d_to_rotation_matrix(pose)
        if pose_size(pose) == 9:
            pose = pose.reshape(-1, 3, 3)
            return _rotation_matrix_to_euler_pyr(pose) if use_pyr_format else _rotation_matrix_to_euler(pose)
    elif to == 'quaternion':
        if pose_size(pose) == 3:
            pose = _euler_to_rotation_matrix_pyr(pose) if use_pyr_format else _euler_to_rotation_matrix(pose)
        if pose_size(pose) == 6:
            pose = _ortho6d_to_rotation_matrix(pose)
        if pose_size(pose) == 9:
            pose = pose.reshape(3, 3)
            return _rotation_matrix_to_quaternion(pose)
    elif to == 'ortho6d':
        if pose_size(pose) == 3:
            pose = _euler_to_rotation_matrix_pyr(pose) if use_pyr_format else _euler_to_rotation_matrix(pose)
        if pose_size(pose) == 4:
            pose = _quaternion_to_rotation_matrix(pose)
        if pose_size(pose) == 9:
            return pose.view(-1, 3, 3)[:, :, :2].transpose(1, 2).reshape(-1, 6)
    elif to == 'matrix':
        if pose_size(pose) == 3:
            return _euler_to_rotation_matrix_pyr(pose) if use_pyr_format else _euler_to_rotation_matrix(pose)
        if pose_size(pose) == 4:
            return _quaternion_to_rotation_matrix(pose)
        if pose_size(pose) == 6:
            return _ortho6d_to_rotation_matrix(pose)
    else:
        raise NotImplementedError(f"Argument 'to' has an invalid value: {to}. "
                                  "Accepted values are 'euler', 'quaternion', 'ortho6d', 'matrix'")
    return pose


# H. Hsu et al., "QuatNet: Quaternion-Based Head Pose Estimation With Multiregression Loss"
def axis_angle_to_quaternion(x):
    assert x.shape[-1] == 4, f"Error: expected (batch_dim, 4) input shape but got: {x.shape}"
    qw = torch.cos(x[:, 3])
    qx = x[:, 0] * torch.sin(x[:, 3])
    qy = x[:, 1] * torch.sin(x[:, 3])
    qz = x[:, 2] * torch.sin(x[:, 3])
    return torch.cat([qw, qx, qy, qz], dim=0)


def _euler_to_rotation_matrix(headpose):
    # http://euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
    # Change coordinates system
    euler = torch.zeros_like(headpose)
    euler[:, 0] = -(headpose[:, 0] - 90)
    euler[:, 1] = -headpose[:, 1]
    euler[:, 2] = -(headpose[:, 2] + 90)

    # Convert to radians
    rad = torch.deg2rad(euler)
    cy = torch.cos(rad[:, 0])
    sy = torch.sin(rad[:, 0])
    cp = torch.cos(rad[:, 1])
    sp = torch.sin(rad[:, 1])
    cr = torch.cos(rad[:, 2])
    sr = torch.sin(rad[:, 2])

    # Init R matrix tensors
    working_device = None
    if euler.is_cuda:
        working_device = euler.device
    Ry = torch.zeros((euler.shape[0], 3, 3), device=working_device)
    Rp = torch.zeros((euler.shape[0], 3, 3), device=working_device)
    Rr = torch.zeros((euler.shape[0], 3, 3), device=working_device)

    # Yaw
    Ry[:, 0, 0] = cy
    Ry[:, 0, 2] = sy
    Ry[:, 1, 1] = 1.
    Ry[:, 2, 0] = -sy
    Ry[:, 2, 2] = cy

    # Pitch
    Rp[:, 0, 0] = cp
    Rp[:, 0, 1] = -sp
    Rp[:, 1, 0] = sp
    Rp[:, 1, 1] = cp
    Rp[:, 2, 2] = 1.

    # Roll
    Rr[:, 0, 0] = 1.
    Rr[:, 1, 1] = cr
    Rr[:, 1, 2] = -sr
    Rr[:, 2, 1] = sr
    Rr[:, 2, 2] = cr

    # For one single matrix
    # Ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])  # yaw
    # Rp = torch.tensor([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])  # pitch
    # Rr = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])  # roll
    return torch.matmul(torch.matmul(Ry, Rp), Rr)


def _euler_to_rotation_matrix_pyr(headpose):
    # Convert to radians
    rad = torch.deg2rad(headpose)
    cy = torch.cos(rad[:, 0])
    sy = torch.sin(rad[:, 0])
    cp = torch.cos(rad[:, 1])
    sp = torch.sin(rad[:, 1])
    cr = torch.cos(rad[:, 2])
    sr = torch.sin(rad[:, 2])

    # Init R matrix tensors
    working_device = None
    if headpose.is_cuda:
        working_device = headpose.device
    Ry = torch.zeros((headpose.shape[0], 3, 3), device=working_device)
    Rp = torch.zeros((headpose.shape[0], 3, 3), device=working_device)
    Rr = torch.zeros((headpose.shape[0], 3, 3), device=working_device)

    # Yaw
    Ry[:, 0, 0] = cy
    Ry[:, 0, 2] = -sy
    Ry[:, 1, 1] = 1.
    Ry[:, 2, 0] = sy
    Ry[:, 2, 2] = cy

    # Pitch
    Rp[:, 0, 0] = 1.
    Rp[:, 1, 1] = cp
    Rp[:, 1, 2] = sp
    Rp[:, 2, 1] = -sp
    Rp[:, 2, 2] = cp

    # Roll
    Rr[:, 0, 0] = cr
    Rr[:, 0, 1] = sr
    Rr[:, 1, 0] = -sr
    Rr[:, 1, 1] = cr
    Rr[:, 2, 2] = 1.

    # For one single matrix
    # Ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])  # yaw
    # Rp = torch.tensor([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])  # pitch
    # Rr = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])  # roll
    return torch.matmul(torch.matmul(Rr, Ry), Rp)


# Y. Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
# https://github.com/papagina/RotationContinuity
def _ortho6d_to_rotation_matrix(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    eps = torch.finfo(poses.dtype).eps
    x = F.normalize(x_raw, dim=1, eps=eps)  # batch*3
    z = torch.cross(x, y_raw, dim=1)        # batch*3
    z = F.normalize(z, dim=1, eps=eps)      # batch*3
    y = torch.cross(z, x, dim=1)            # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def _quaternion_to_rotation_matrix(quaternion):
    batch = quaternion.shape[0]

    quat = F.normalize(quaternion, dim=1, eps=torch.finfo(quaternion.dtype).eps)

    qw = quat[..., 0].view(batch, 1)
    qx = quat[..., 1].view(batch, 1)
    qy = quat[..., 2].view(batch, 1)
    qz = quat[..., 3].view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


def _rotation_matrix_to_euler(rot_matrix):
    # http://euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
    a00, a02 = rot_matrix[:, 0, 0], rot_matrix[:, 0, 2]
    a10, a11, a12 = rot_matrix[:, 1, 0], rot_matrix[:, 1, 1], rot_matrix[:, 1, 2]
    a20, a22 = rot_matrix[:, 2, 0], rot_matrix[:, 2, 2]

    # Standard case
    yaw = torch.atan2(-a20, a00)
    pitch = torch.arcsin(a10)
    roll = torch.atan2(-a12, a11)

    eps = torch.finfo(rot_matrix.dtype).eps
    s_north = torch.abs(1.0 - a10) <= eps  # singularity at north pole / special case a10 == 1
    s_south = torch.abs(-1.0 - a10) <= eps  # singularity at south pole / special case a10 == -1

    yaw[s_north | s_south] = torch.atan2(a02, a22)[s_north | s_south]
    pitch[s_north] = math.pi / 2
    pitch[s_south] = -math.pi / 2
    roll[s_north | s_south] = 0

    # Convert to degrees
    euler = torch.rad2deg(torch.vstack((yaw, pitch, roll))).T
    # Change coordinates system
    euler[:, 0] = (-euler[:, 0]) + 90
    euler[:, 1] = -euler[:, 1]
    euler[:, 2] = (-euler[:, 2]) - 90
    euler[euler < -180] = euler[euler < -180] + 360
    euler[euler > 180] = euler[euler > 180] - 360
    return euler


def _rotation_matrix_to_euler_pyr(rot_matrix):
    # http://euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
    a00, a01, a02 = rot_matrix[:, 0, 0], rot_matrix[:, 0, 1], rot_matrix[:, 0, 2]
    a10, a11, a12 = rot_matrix[:, 1, 0], rot_matrix[:, 1, 1], rot_matrix[:, 1, 2]
    a20, a21, a22 = rot_matrix[:, 2, 0], rot_matrix[:, 2, 1], rot_matrix[:, 2, 2]

    # Standard case
    yaw = torch.asin(a20)
    pitch = -torch.atan2(a21, a22)
    roll = -torch.atan2(a10, a00)

    eps = torch.finfo(rot_matrix.dtype).eps
    s_north = torch.abs(1.0 - a20) <= eps  # singularity at north pole / special case a10 == 1
    s_south = torch.abs(-1.0 - a20) <= eps  # singularity at south pole / special case a10 == -1

    pitch[s_north | s_south] = torch.atan2(a12, a11)[s_north | s_south]
    yaw[s_north] = math.pi / 2
    yaw[s_south] = -math.pi / 2
    roll[s_north | s_south] = 0

    # Convert to degrees
    euler = torch.rad2deg(torch.vstack((yaw, pitch, roll))).T
    euler[euler < -180] = euler[euler < -180] + 360
    euler[euler > 180] = euler[euler > 180] - 360
    return euler


def _rotation_matrix_to_quaternion(rot_matrix):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    m00, m01, m02 = rot_matrix[:, 0, 0], rot_matrix[:, 0, 1], rot_matrix[:, 0, 2]
    m10, m11, m12 = rot_matrix[:, 1, 0], rot_matrix[:, 1, 1], rot_matrix[:, 1, 2]
    m20, m21, m22 = rot_matrix[:, 2, 0], rot_matrix[:, 2, 1], rot_matrix[:, 2, 2]

    quaternion = torch.zeros((rot_matrix.shape[0], 4), dtype=rot_matrix.dtype, device=rot_matrix.device)

    tr = m00 + m11 + m22
    mask_1 = tr > 0
    mask_2 = (m00 > m11) & (m00 > m22)
    mask_3 = (m11 > m22)
    mask_4 = ~(mask_1 | mask_2 | mask_3)
    S_1 = torch.sqrt(tr + 1.0) * 2  # S=4*qw
    S_2 = torch.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*qx
    S_3 = torch.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*qy
    S_4 = torch.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*qz

    quaternion[mask_1] = torch.vstack([0.25 * S_1,
                                       (m21 - m12) / S_1,
                                       (m02 - m20) / S_1,
                                       (m10 - m01) / S_1]).T[mask_1]
    quaternion[mask_2] = torch.vstack([(m21 - m12) / S_2,
                                       0.25 * S_2,
                                       (m01 + m10) / S_2,
                                       (m02 + m20) / S_2]).T[mask_2]
    quaternion[mask_3] = torch.vstack([(m02 - m20) / S_3,
                                       (m01 + m10) / S_3,
                                       0.25 * S_3,
                                       (m12 + m21) / S_3]).T[mask_3]
    quaternion[mask_4] = torch.vstack([(m10 - m01) / S_4,
                                       (m02 + m20) / S_4,
                                       (m12 + m21) / S_4,
                                       0.25 * S_4]).T[mask_4]
    quaternion[quaternion[:, 0] < 0, :] *= -1
    return quaternion
