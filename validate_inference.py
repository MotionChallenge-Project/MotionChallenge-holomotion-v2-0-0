"""Compare CPU adaptation with the pinned author CUDA FK/observation and runtime.

uv run --with torch --with warp-lang --with pyyaml validate_inference.py
  --source /path/to/HoloMotion --weights /path/to/model_10500.onnx --output REPORT
CUDA is required for this audit only. Evaluation itself runs on CPU.
"""
import argparse
import ast
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import xml.etree.ElementTree as ETree

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import yaml

from adapter import HoloMotion, rot6_xz
from motion_challenge.physics import World, forward_kinematics
from motion_challenge.sdk import ReferenceFrame


def definitions(path, scope, names, methods=None):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    if methods is not None:
        for node in nodes:
            node.body = [m for m in node.body if isinstance(m, ast.FunctionDef) and m.name in methods]
            assert {m.name for m in node.body} == set(methods)
    exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+nodes, []),
                 str(path), 'exec'), scope)


def native_modules(source):
    # Namespace/import glue only: source functions, kernels and MJCF parser are unchanged.
    for name, folder in [('holomotion', 'holomotion'), ('holomotion.src', 'holomotion/src'),
                         ('holomotion.src.training', 'holomotion/src/training'),
                         ('holomotion.src.motion_tracking', 'holomotion/src/motion_tracking'),
                         ('humanoid_policy', 'deployment/unitree_g1_ros2_29dof/src/humanoid_policy')]:
        module = ModuleType(name); module.__path__ = [str(source/folder)]
        sys.modules[name] = module
    module = ModuleType('holomotion.src.training.reference_fk')
    module.__dict__.update(torch=torch, os=os, ETree=ETree)
    definitions(source/'holomotion/src/training/reference_fk.py', module.__dict__, ['MJCFParser', 'URDFParser'])
    sys.modules[module.__name__] = module
    return (importlib.import_module('humanoid_policy.observation').PolicyObservationBuilder,
            importlib.import_module('humanoid_policy.vr_reference_adapter').VRJointReferenceAdapter)


def main():
    parser = argparse.ArgumentParser()
    for key in ('source', 'weights', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    pins = json.loads((root/'native-sources.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, digest in pins['files'].items():
        assert sha(args.source/name) == digest, name
    assert sha(root/'validation/author-config.yaml') == pins['checkpoint_config_sha256']
    config = yaml.safe_load((root/'validation/author-config.yaml').read_text())
    runtime = json.loads((root/'runtime.json').read_text()); runtime['weights'] = str(args.weights.resolve())
    policy = HoloMotion(root, runtime)
    metadata = policy.session.get_modelmeta().custom_metadata_map
    assert metadata == pins['onnx_metadata']
    assert policy.names == metadata['joint_names'].split(',')
    for attr, key in [('default', 'default_joint_pos'), ('scale', 'action_scale'),
                      ('kp', 'joint_stiffness'), ('kd', 'joint_damping')]:
        np.testing.assert_array_equal(getattr(policy, attr), np.array(metadata[key].split(','), np.float32))
    links = config['link_observation']
    assert links['schema_version'] == 'g1_link_xz6d_v1'
    assert links['link_names'] == policy.links
    assert links['frame_offsets'] == [1, 2, 3, 4, 5]
    assert links['mask']['mode_candidates']['WholeBody-14'] == policy.links
    assert links['include_error_terms'] and not links.get('include_future_link_mask', False)
    assert config['obs']['context_length'] == 1 and config['obs']['target_fps'] == 50
    assert runtime['control_period_s'] == .02 and runtime['reference_offsets_s'] == [0., .02, .04, .06, .08, .1]
    Builder, Reference = native_modules(args.source)
    asset = args.source/'assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.xml'
    contract = dict(asset_sha256=sha(asset), include_error_terms=True, include_future_link_mask=False,
        frame_offsets=links['frame_offsets'], link_names=policy.links, actor_dim=1378,
        modes={'WholeBody-14': [1]*14}, config_joint_names=config['robot']['dof_names'],
        onnx_index_for_config_joint=[policy.names.index(n) for n in config['robot']['dof_names']],
        default_joint_pos_onnx=policy.default.astype(np.float32))
    builder = Builder(contract, asset, device=args.device)
    world = World(root/'motion-challenge/robots/g1-29dof-cpu-v0.json')
    rng = np.random.default_rng(241025)
    shuffle = rng.permutation(29)
    names = tuple(np.asarray(world.joint_names)[shuffle])
    reference = Reference(contract, asset, names, device=args.device)
    native = args.source/'deployment/unitree_g1_ros2_29dof/src/humanoid_policy'
    scope = {'np': np}
    definitions(native/'offline_motion_reference.py', scope, ['yaw_from_quat_wxyz'])
    definitions(native/'observation_evaluator.py', scope, ['PolicyObservationEvaluator'],
        {'begin_motion_yaw_alignment', '_standardize_quaternion_wxyz', '_yaw_quat_wxyz', 'clear_motion_yaw_alignment'})
    definitions(native/'policy_runtime.py', scope, ['PolicyRuntime'],
        {'reset_counters', '_maybe_reset_motion_rope_window', '_run_current_policy', 'apply_policy_output'})
    log = NS(info=lambda *_: None, warn=lambda *_: None)
    errors = {k: 0. for k in ('heading', 'measured_fk_position', 'measured_fk_rotation', 'observation',
                             'network', 'action', 'motor_target', 'cache', 'reset_cache', 'kp', 'kd')}
    def compare(key, a, b):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    def sample(index):
        rotation = R.from_euler('xyz', [.1, np.pi/2-1e-5, -.5]) if index % 10 == 7 else R.random(random_state=rng)
        world.data.qpos[:] = np.r_[rng.normal(0, .4, 2), .8, rotation.as_quat()[[3, 0, 1, 2]], rng.normal(0, .3, 29)]
        world.data.qvel[:] = rng.normal(0, .5, 35)
        forward_kinematics(world.model, world.data)
        state = world.state()
        return replace(state, joint_names=names,
            joint_position_rad=tuple(np.asarray(state.joint_position_rad)[shuffle]),
            joint_velocity_rad_s=tuple(np.asarray(state.joint_velocity_rad_s)[shuffle]))
    def frame(state, index):
        return ReferenceFrame(index*.02, index*.02, state.joint_position_rad, state.root_position_world_m,
            state.root_orientation_xyzw, state.joint_velocity_rad_s, state.root_linear_velocity_world_m_s,
            state.root_angular_velocity_world_rad_s, state.bodies)
    cases = endpoint_cases = 0
    for reset in range(4):
        port = NS(use_kv_cache=True, motion_kv_cache=np.zeros((1, 2, 1, 32, 4, 64), np.float32),
            motion_kv_shape=[1, 2, 1, 32, 4, 64], motion_kv_dtype=np.float32,
            motion_input_name='obs', motion_output_name='actions', motion_kv_input_name='past_key_values',
            motion_kv_output_name='present_key_values', motion_step_idx_input_name='step_idx',
            motion_policy_session=policy.session, motion_rope_max_seq_len=int(metadata['rope_max_seq_len']),
            motion_rope_reset_margin=64, get_logger=lambda: log)
        engine = scope['PolicyRuntime'](); engine.port = port
        engine.state = NS(current_policy_mode='motion', motion_step_idx=0, motion_frame_idx=0,
            actions_onnx=np.zeros(29, np.float32), target_dof_pos_onnx=np.zeros(29, np.float32),
            target_dof_pos_real_buffers=[np.zeros(29, np.float32), np.zeros(29, np.float32)], target_dof_pos_real_buffer_idx=0)
        engine._record_reset_epoch = 0
        engine._motion_input_feed = engine._motion_output_names = engine._motion_step_idx_array = None
        engine.reset_counters()
        for step in range(40):
            state = sample(step)
            refs = [frame(sample(step+i+1), i) for i in range(6)]
            if step >= 35:
                refs[3:] = [replace(refs[2], requested_time_s=i*.02) for i in range(3, 6)]
                endpoint_cases += 1
            if step == 0:
                policy.reset(state, refs, reset)
                observer = scope['PolicyObservationEvaluator'].__new__(scope['PolicyObservationEvaluator'])
                observer.robot_root_rot_quat_wxyz = np.asarray(state.root_orientation_xyzw)[[3, 0, 1, 2]]
                observer._get_ref_current_root_quat_wxyz = lambda: np.asarray(refs[0].root_orientation_xyzw)[[3, 0, 1, 2]]
                observer.get_logger = lambda: log
                observer.begin_motion_yaw_alignment()
                alignment = observer._motion_ref_yaw_alignment_quat_wxyz
                compare('heading', policy.heading.as_matrix(), R.from_quat(alignment[[1, 2, 3, 0]]).as_matrix())
                compare('reset_cache', policy.memory, port.motion_kv_cache)
            if step == 37:
                policy.step = engine.state.motion_step_idx = 8127
            sequence = np.array([np.r_[r.root_position_world_m,
                np.asarray(r.root_orientation_xyzw)[[3, 0, 1, 2]], r.joint_position_rad]
                for r in [refs[0], *refs, refs[-1]]], np.float32)
            command = reference.from_host_sequence(sequence, yaw_alignment_wxyz=alignment)
            actual = R.from_quat(state.root_orientation_xyzw)
            order = [names.index(n) for n in policy.names]
            native_obs = builder.build(mode='WholeBody-14', reference=command,
                robot_root_quat_wxyz=np.asarray(state.root_orientation_xyzw)[[3, 0, 1, 2]],
                robot_dof_pos_onnx=np.asarray(state.joint_position_rad)[order],
                robot_dof_vel_onnx=np.asarray(state.joint_velocity_rad_s)[order],
                projected_gravity=actual.inv().apply([0, 0, -1]),
                root_ang_vel_local=actual.inv().apply(state.root_angular_velocity_world_rad_s),
                last_action_onnx=engine.state.actions_onnx).array.numpy()
            lookup = {body.name: body for body in state.bodies}
            bodies = [lookup[n] for n in policy.links]
            indices = [builder.fk.body_names.index(n) for n in policy.links]
            compare('measured_fk_position', actual.inv().apply(np.asarray([b.position_world_m for b in bodies])-state.root_position_world_m),
                    builder.fk.positions.numpy()[0, indices])
            compare('measured_fk_rotation', rot6_xz(actual.inv()*R.from_quat([b.orientation_xyzw for b in bodies])),
                    rot6_xz(R.from_quat(builder.fk.orientations.numpy()[0, indices])))
            inputs = policy.observations(state, refs)
            compare('observation', inputs['obs'], native_obs)
            expected = engine._run_current_policy(native_obs)
            compare('network', policy.infer(engine._motion_input_feed)['actions'], expected[0])
            target = engine.apply_policy_output(expected[0], action_scale=policy.scale.astype(np.float32),
                default_angles=policy.default.astype(np.float32), onnx_to_real=policy.inverse, is_motion=True)
            method = policy.observations; policy.observations = lambda *_: inputs
            result = policy.act(state, refs); policy.observations = method
            compare('action', policy.last, engine.state.actions_onnx)
            compare('motor_target', result.joint_position_target_rad, target)
            compare('cache', policy.memory, port.motion_kv_cache)
            compare('kp', result.kp_nm_rad, policy.kp[policy.inverse])
            compare('kd', result.kd_nm_s_rad, policy.kd[policy.inverse])
            assert policy.step == engine.state.motion_step_idx
            cases += 1
        print(json.dumps({'reset': reset+1, 'cases': cases, 'max_abs_error': errors}), flush=True)
    limits = dict(heading=2e-4, measured_fk_position=2e-4, measured_fk_rotation=2e-4,
        observation=2e-4, network=0., action=2e-3, motor_target=2e-3, cache=2e-3, reset_cache=0., kp=0., kd=0.)
    report = dict(passed=all(errors[k] <= limits[k] for k in errors), cases=cases, resets=4,
        endpoint_cases=endpoint_cases, rope_boundary=8128, max_abs_error=errors, limits=limits,
        checkpoint_sha256=runtime['weights_sha256'], source_commit=pins['source_commit'], version=pins['version'],
        scope='Original CUDA Warp reference/measured FK and 1378-D observation; original motion-entry heading and CPU ONNX recurrent runtime; motor targets, joint permutations, endpoint clamping and cache reset. Canonical physics unchanged; GPU ONNX equivalence is separate.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2)); assert report['passed']


if __name__ == '__main__':
    main()
