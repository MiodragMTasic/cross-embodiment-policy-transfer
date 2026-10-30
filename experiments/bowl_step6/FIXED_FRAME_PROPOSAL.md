# Fixed adapter frame proposal

The approved native diagnostics provide current camera poses and authored gripper geometry. This proposal changes no model weights or mesh/physics properties.

Use a common gripping coordinate system: +X points out of the fingers, +Y points toward the named left finger, and +Z completes the right-handed frame. The origin is the mean distal plane of the two open fingertip pads, on the mechanical symmetry axis. Keep that origin fixed when the gripper closes. A's approach/closing axes are raw +X/+Y; B's are raw +Z/+X. These statements follow the measured named meshes, not the spelling of joint names. The two gripper models have small dimensional differences that this rigid mapping does not eliminate.

Let C_A and C_B map canonical coordinates into each robot's raw tool frame. To express A's requested raw tool pose on B, use T_B = T_A C_A inverse(C_B). Convert through the measured robot-root world transforms before IK. The observation direction is the inverse mapping. Never treat equal-length joint vectors as equivalent.

Mount B's simulated wrist camera on its raw gripper base, with offset C_B inverse(C_A) T_A_camera. Copy A's resolution and intrinsics. This is a declared camera remount in simulation, part of the fixed interface; it is neither image synthesis nor a claim that the physical Kinova camera has this pose. The exterior camera already has the same pose/intrinsics; verify it numerically. The later live test must verify actual camera pose/intrinsics and rendered views after the remount.

Before physical execution, perform 16 matrix/camera equivalence checks (eight phases in each mapping direction). Then run a fixed 32-case CPU IK bank: matched A home plus every tenth row of the previously measured 301-row A development trajectory. Seed B from its native home, then continue from each previous successful solution. Reverse-map successful B poses to virtual A, also continuing from previous success. Every solve is capped at 200 iterations, with no retry or hidden branch search; preserve failures. The existing development trajectory is training-route case 2002, never validation/final/FGH data. It is used to diagnose feasible poses, not to fit parameters or report transfer success.

This does not establish collision avoidance, contact behavior, equivalent camera images, a working policy, or readiness to train P1. A-only training and all held-out experiments remain gated.
