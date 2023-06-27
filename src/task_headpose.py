import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseHead(nn.Module):

    def __init__(self,
                 model,
                 head_name='pose',
                 rotation_mode='euler'):

        super(PoseHead, self).__init__()

        self.head_name = head_name
        self.head_format = head_name + '_%i'
        self.in_core = model.channels_core
        self.out_cores = model.out_cores

        self.rotation_mode = rotation_mode

        # Fully-connected layer
        if rotation_mode == 'euler':
            channels_pose = 3
        elif rotation_mode == 'quaternion':
            channels_pose = 4
        elif rotation_mode == '6d' or rotation_mode == '6d_opal':
            channels_pose = 6
        else:
            raise NotImplementedError(f"Unknown rotation mode: {rotation_mode}")

        self.fc_for_pose = nn.ModuleDict()
        for i_core in range(self.out_cores):
            fc_name = self.head_format % i_core
            self.fc_for_pose[fc_name] = nn.Linear(self.in_core, channels_pose)

    def forward(self, raw_output):

        pose_in = raw_output['core']
        pose_out = []
        for i_core in range(self.out_cores):
            pose = pose_in[i_core]
            pose = pose.reshape((pose.shape[:2]))
            fc_name = self.head_format % i_core
            pose = self.fc_for_pose[fc_name](pose)

            if self.rotation_mode == 'quaternions':
                pose = self._axis_angle_to_quaternion(pose)
            elif self.rotation_mode == '6d' or self.rotation_mode == '6d_opal':
                pose = self._6d_to_rotation_matrix(pose)

            pose = pose.reshape(pose.shape[0], -1)
            pose_out.append(pose)

        # Save pose output
        return pose_out[-1]

    # H. Hsu et al., "QuatNet: Quaternion-Based Head Pose Estimation With Multiregression Loss"
    def _axis_angle_to_quaternion(self, x):
        assert x.shape[-1] == 4, f"Error: expected (batch_dim, 4) input shape but got: {x.shape}"
        qw = torch.cos(x[:, 3])
        qx = x[:, 0] * torch.sin(x[:, 3])
        qy = x[:, 1] * torch.sin(x[:, 3])
        qz = x[:, 2] * torch.sin(x[:, 3])
        return torch.cat([qw, qx, qy, qz], dim=0)

    # Y. Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    # https://github.com/papagina/RotationContinuity
    def _6d_to_rotation_matrix(self, poses):
        x_raw = poses[:, 0:3]  # batch*3
        y_raw = poses[:, 3:6]  # batch*3

        eps = torch.finfo(poses.dtype).eps
        x = F.normalize(x_raw, dim=1, eps=eps)  # batch*3
        z = torch.cross(x, y_raw, dim=1)  # batch*3
        z = F.normalize(z, dim=1, eps=eps)  # batch*3
        y = torch.cross(z, x, dim=1)  # batch*3

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        matrix = torch.cat((x, y, z), 2)  # batch*3*3
        return matrix
