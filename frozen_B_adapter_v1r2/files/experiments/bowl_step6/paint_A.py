"""Exterior-only cross-painting for P0 on B (Mirage-style, simulator ground truth).

A visual-only copy of robot A (the DROID Franka + Robotiq 2F-85 asset, contact reporting off, collisions disabled where
the asset allows, gravity off) is added to B's scene, parked 20 m above the cell whenever physics runs. Before an
exterior render that P0 will receive, its root is teleported to A's root pose and its joints to the virtual-A joints the observation bridge already computes from B's measured state, its gripper is set from the
same closure P0 receives, B's visuals are hidden and A's are shown. After the render both are restored and the copy is parked again.
The copy and B never overlap during a physics step (probe v1 showed the asset's instanced colliders stay active).

Simulator information used: B's measured joint state (through the existing virtual-A IK), B's gripper knuckle angle,
and the simulator's renderer/scene. Nothing is learned. Physics is untouched: the copy is never inside the cell during a
physics step, and only its root/joint state is written. The wrist camera image is never painted.
"""
import numpy as np
import torch

EXTERIOR = 'over_shoulder_left_camera'
ARM = [f'panda_joint{i}' for i in range(1, 8)]
# Mimic signs of the six Robotiq joints, read from A2002 run_0 (row 120: +g,+g,-g,+g,-g,-g with g = finger_joint).
GRIPPER_SIGNS = {'finger_joint': 1., 'right_outer_knuckle_joint': 1., 'left_inner_finger_joint': -1.,
                 'right_inner_finger_joint': 1., 'left_inner_finger_knuckle_joint': -1.,
                 'right_inner_finger_knuckle_joint': -1.}
A_FINGER_FULL_RAD = np.pi / 4  # A gripper observation = finger_joint / (pi/4) in A2002 (ratio verified 1.2732)
PARK = (0., 0., 20.)  # world position of the copy's root whenever physics steps
RENDER_FRAMES = 4  # renders after each visibility switch; set from the development probe, then frozen


def add_painted_A(cfg):
    """Add the collision-free visual copy of A. Call before create_env, in both painted and unpainted arms."""
    import isaaclab.sim as sim_utils
    from robolab.robots.droid import DroidCfg
    base = DroidCfg().robot
    spawn = base.spawn.replace(activate_contact_sensors=False, visible=False,
                               collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
    cfg.scene.painted_A = base.replace(prim_path='{ENV_REGEX_NS}/painted_A', spawn=spawn,
                                       init_state=base.init_state.replace(pos=PARK))
    return cfg


class Painter:
    def __init__(self, env, frames=RENDER_FRAMES):
        from pxr import Usd, UsdGeom
        import omni.usd
        self.env, self.frames = env, frames
        self.A = env.scene['painted_A']
        names = list(self.A.data.joint_names)
        self.arm = [names.index(n) for n in ARM]
        self.grip = [(names.index(n), s) for n, s in GRIPPER_SIGNS.items()]
        stage = omni.usd.get_context().get_stage()
        ns = env.scene.env_prim_paths[0]
        self.A_root = stage.GetPrimAtPath(ns + '/painted_A')
        B_root = stage.GetPrimAtPath(ns + '/robot')
        assert self.A_root.IsValid() and B_root.IsValid()
        # Hide B's render geometry only: 'visuals' groups (instance roots) where present, else leaf Gprims.
        # Cameras and their ancestors are never touched, so the wrist camera prim keeps its visibility.
        visuals = [p for p in Usd.PrimRange(B_root) if p.GetName() == 'visuals' and p.IsA(UsdGeom.Imageable)]
        if not visuals:
            visuals = [p for p in Usd.PrimRange(B_root) if p.IsA(UsdGeom.Gprim)]
        self.B_visuals = [UsdGeom.Imageable(p) for p in visuals]
        self.A_image = UsdGeom.Imageable(self.A_root)
        assert self.B_visuals, 'no B render geometry found'
        self.camera = env.scene[EXTERIOR]
        origin = env.scene.env_origins[0].detach().cpu().numpy()
        self.at_A = torch.tensor([[*origin, 1., 0., 0., 0.]], device=env.device)  # A root = env origin, identity (frame_proposal)
        self.parked = torch.tensor([[*(origin + np.asarray(PARK)), 1., 0., 0., 0.]], device=env.device)
        self.inventory = {'A_root': str(self.A_root.GetPath()), 'B_hidden_prims': [str(p.GetPrim().GetPath()) for p in self.B_visuals],
                          'render_frames_per_switch': frames, 'A_joint_names': names}

    def _show(self, painted):
        for p in self.B_visuals:
            (p.MakeInvisible if painted else p.MakeVisible)()
        (self.A_image.MakeVisible if painted else self.A_image.MakeInvisible)()

    def _render(self):
        for _ in range(self.frames):
            self.env.sim.render()
        self.camera.update(0.0, force_recompute=True)
        return self.camera.data.output['rgb'][0, ..., :3].detach().cpu().numpy().copy()

    def pose(self, virtual_A_q, closure):
        q = self.A.data.default_joint_pos.clone()
        q[0, self.arm] = torch.as_tensor(np.asarray(virtual_A_q, float), device=q.device, dtype=q.dtype)
        g = float(np.clip(closure, 0, 1)) * A_FINGER_FULL_RAD
        for i, s in self.grip:
            q[0, i] = s * g
        self.A.write_root_pose_to_sim(self.at_A)
        self.A.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.A.set_joint_position_target(q)
        self.A.write_data_to_sim()
        return q[0].detach().cpu().numpy().copy()

    def park(self):
        self.A.write_root_pose_to_sim(self.parked)
        self.A.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device))
        self.env.sim.forward()

    def paint(self, virtual_A_q, closure):
        """Return (painted exterior RGB, written A joint vector). Scene visibility is restored before returning."""
        written = self.pose(virtual_A_q, closure)
        self._show(True)
        try:
            painted = self._render()
        finally:
            self.park()
            self._show(False)
            self._render()
        return painted, written
