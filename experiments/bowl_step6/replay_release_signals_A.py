"""Unchanged A release-signal extraction copied from reviewed step5 v4 instrumentation."""
import numpy as np
from robolab.core.world.world_state import get_world
from robolab.core.sensors.contact_sensor_utils import get_contact_sensor_with_order
from robolab.core.task.conditionals import object_in_container
from rim_support import support_signals

def release_signals(env):
    world = get_world(env)
    boxes = {n: world.get_bbox(n)[0][0].detach().cpu().numpy() for n in ('banana', 'bowl', 'table')}
    sensors = {n: get_contact_sensor_with_order(env.scene, 'banana', n) for n in ('gripper', 'bowl', 'table')}
    forces = {n: sensor.data.force_matrix_w[0].detach().cpu().numpy() for n, (sensor, _) in sensors.items()}
    support = support_signals(boxes['banana'], boxes['bowl'], forces['bowl'], sensors['bowl'][1])
    idx = env.scene['robot'].data.joint_names.index('finger_joint')
    closure = float(env.scene['robot'].data.joint_pos[0, idx].item()) / (np.pi/4)
    poses = {n: [x.detach().cpu().tolist() for x in world.get_pose(n, env_id=0)]
             for n in ('banana', 'bowl', 'table')}
    finite = bool(all(np.isfinite(v).all() for v in list(boxes.values()) + list(forces.values()))
                  and np.isfinite(closure) and all(np.isfinite(x).all() for v in poses.values() for x in v))
    contacts = {n: bool(np.any(np.abs(f) > .1)) for n, f in forces.items()}
    clearance = float(boxes['banana'][:, 2].min() - boxes['table'][:, 2].max())
    height_above_bowl = float(boxes['banana'][:, 2].mean() - boxes['bowl'][:, 2].max())
    geometric = bool(object_in_container(env, object='banana', container='bowl',
                      require_contact_with=False, require_gripper_detached=False, env_id=0)) if finite else False
    return {**support, 'finite': finite, 'contact': contacts['gripper'], 'table_contact': contacts['table'],
            'bowl_contact': contacts['bowl'], 'closure': closure, 'clearance_m': clearance,
            'near_bowl': geometric and height_above_bowl <= .15,
            'geometric_bowl_predicate': geometric, 'centroid_height_above_bowl_m': height_above_bowl,
            'boxes_env_local': {k: v.tolist() for k,v in boxes.items()},
            'forces_world': {k:v.tolist() for k,v in forces.items()},
            'force_sensor_reversed': {k: bool(v[1]) for k,v in sensors.items()}, 'poses_env_local': poses}
