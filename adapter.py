"""HoloMotion v2 WholeBody-14 controller with the released recurrent ONNX."""
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation as R

from motion_challenge.adapters.onnx import ONNXController, yaw


def rot6_xz(rotation):
    matrix = rotation.as_matrix()
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 2]), axis=-1)


class HoloMotion(ONNXController):
    action_output = 'actions'

    def __init__(self, root, runtime):
        super().__init__(root, runtime)
        if self.session.get_inputs()[0].shape != [1, 1378]:
            raise ValueError('expected the released v2 WholeBody-14 input schema')
        self.links = self.cfg['link_names']

    def reset(self, state, reference, seed):
        super().reset(state, reference, seed)
        self.memory = np.zeros((1, 2, 1, 32, 4, 64), np.float32)
        self.step = 0
        heading = yaw(R.from_quat(state.root_orientation_xyzw))-yaw(R.from_quat(reference[0].root_orientation_xyzw))
        self.heading = R.from_rotvec([0., 0., float(heading)])

    def observations(self, state, reference):
        q, qd, omega, gravity, actual = self.proprio(state)
        measured = {body.name: body for body in state.bodies}
        bodies = [measured[name] for name in self.links]
        local_position = actual.inv().apply(np.asarray([b.position_world_m for b in bodies])-state.root_position_world_m)
        local_rotation = actual.inv()*R.from_quat([b.orientation_xyzw for b in bodies])
        current_root = np.asarray(reference[0].root_position_world_m)
        target_positions, position_errors, target_rotations, rotation_errors = [], [], [], []
        for frame in reference[1:]:
            lookup = {body.name: body for body in frame.bodies}
            target = [lookup[name] for name in self.links]
            # Native motion-entry yaw calibration rotates about the current reference root.
            positions = (actual.inv()*self.heading).apply(np.asarray([b.position_world_m for b in target])-current_root)
            rotations = actual.inv()*self.heading*R.from_quat([b.orientation_xyzw for b in target])
            target_positions.append(positions)
            position_errors.append(positions-local_position)
            target_rotations.append(rot6_xz(rotations))
            rotation_errors.append(rot6_xz(rotations*local_rotation.inv()))
        obs = np.concatenate((gravity, omega, q-self.default, qd, self.last,
            current_root[2:3], np.ones(14), np.ones(5),
            np.asarray(target_positions).ravel(), np.asarray(position_errors).ravel(),
            np.asarray(target_rotations).ravel(), np.asarray(rotation_errors).ravel(),
            np.arange(1, 6))).astype(np.float32)[None]
        if self.step >= self.cfg['rope_reset_at']:
            self.memory.fill(0)
            self.step = 0
        inputs = {'obs': obs, 'past_key_values': self.memory, 'step_idx': np.array([self.step], np.int64)}
        self.step += 1
        return inputs

    def after_inference(self, result):
        self.memory = result['present_key_values']

    def act(self, state, reference):
        command = super().act(state, reference)
        target = self.default.astype(np.float32)+self.scale.astype(np.float32)*self.last
        return replace(command, joint_position_target_rad=tuple(float(v) for v in target[self.inverse]))


def create(root, runtime):
    return HoloMotion(root, runtime)
