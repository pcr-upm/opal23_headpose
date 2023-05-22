import torch.nn as nn

# My libs
from .utils import axis_angle_to_quaternion


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
        elif rotation_mode == 'ortho6d':
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
                pose = axis_angle_to_quaternion(pose)

            pose = pose.reshape(pose.shape[0], -1)
            pose_out.append(pose)

        # Save pose output
        return pose_out[-1]
