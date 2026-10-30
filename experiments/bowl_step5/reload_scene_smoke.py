"""Fresh disposable-checkpoint inference and exactly eight simulation steps.

This checks execution plumbing only. It is not a completed task trial or P1.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path.home() / "robot-transfer/bowl"
REPO = ROOT / "RoboLab"
OUT = ROOT / "runs/step5_reload_smoke_v1/scene"
OUT.mkdir(exist_ok=False)
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ["OMNI_KIT_ACCEPT_EULA"] = "Y"
import cv2
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app
env = client = None
try:
    import numpy as np
    import omni.usd
    import torch
    import zmq
    import robolab.constants
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.task.instruction import primary_instruction
    from policies.gr00t.client import GR00TDroidJointposClient
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
    robolab.constants.set_output_dir(str(OUT))
    auto_register_droid_envs(task=["BananaInBowlTask"])
    names = get_envs(task=["BananaInBowlTask"])
    assert len(names) == 1, names
    env, cfg = create_env(names[0], device="cuda:0", seed=0, num_envs=1,
                          instruction_type=["default"], policy="gr00t")
    obs, _ = env.reset()
    obs, _ = env.reset()
    _, instruction = primary_instruction(cfg.instruction)
    report = {"task": names[0], "instruction": instruction, "seed": 0,
              "purpose": "disposable checkpoint reload and execution only; not task success or P1",
              "policy_actions_executed": 0, "execution_horizon": 8, "images": {},
              "proprioception": {}, "asset_errors": []}
    for name, tensor in obs["image_obs"].items():
        frame = tensor[0].detach().cpu().numpy()
        assert frame.dtype == np.uint8 and frame.ndim == 3, (name, frame.shape, frame.dtype)
        assert frame.std() > 2, (name, "blank camera frame")
        cv2.imwrite(str(OUT / f"{name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        report["images"][name] = {"shape": list(frame.shape), "std": float(frame.std()),
                                  "min": int(frame.min()), "max": int(frame.max())}
    for name, tensor in obs.get("viewport_cam", {}).items():
        frame = tensor[0].detach().cpu().numpy()
        cv2.imwrite(str(OUT / f"viewport_{name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    for name, tensor in obs["proprio_obs"].items():
        arr = tensor.detach().cpu().numpy()
        assert np.isfinite(arr).all(), name
        report["proprioception"][name] = arr.tolist()
    (OUT / "observation-check.json").write_text(json.dumps(report, indent=2) + "\n")
    stage = omni.usd.get_context().get_stage()
    report["asset_errors"] = sorted({
        f"{prim.GetPath()}: {error}"
        for prim in stage.TraverseAll()
        for error in prim.GetPrimIndex().localErrors
    })
    report["used_layers"] = [layer.identifier for layer in stage.GetUsedLayers()]
    assert not report["asset_errors"], report["asset_errors"]
    client = GR00TDroidJointposClient(remote_host="127.0.0.1", remote_port=15555, open_loop_horizon=8)
    client.client.socket.setsockopt(zmq.RCVTIMEO, 180000)
    client.client.socket.setsockopt(zmq.SNDTIMEO, 180000)
    client.client.socket.setsockopt(zmq.LINGER, 0)
    extracted = client._extract_observation(obs)
    request = client._pack_request(extracted, cfg.instruction)
    np.savez_compressed(OUT / "real-observation.npz", **{k:v for k,v in request.items() if isinstance(v,np.ndarray)})
    report["request_shapes"] = {k:list(v.shape) for k,v in request.items() if isinstance(v,np.ndarray)}
    (OUT / "scene-before-inference.json").write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    response = client._query_server(request)
    report["inference_seconds"] = time.perf_counter() - started
    chunk = client._unpack_response(response)
    assert chunk.shape == (40, 8) and np.isfinite(chunk).all(), chunk.shape
    np.save(OUT / "raw-action-chunk.npy", chunk)
    mapped = client._postprocess_chunk(chunk)
    np.save(OUT / "mapped-action-chunk.npy", mapped)
    report["action"] = {"shape": list(chunk.shape), "finite": bool(np.isfinite(chunk).all()),
                        "min": float(chunk.min()), "max": float(chunk.max()),
                        "first_raw": chunk[0].tolist(), "first_mapped": mapped[0].tolist()}
    assert mapped.shape == (40, 8) and np.isfinite(mapped).all()
    state_trace = []
    frames = []
    for row in range(8):
        action = torch.from_numpy(mapped[row].copy()).to(device=env.device, dtype=torch.float32).unsqueeze(0)
        obs, reward, terminated, truncated, info = env.step(action)
        state = {name: value.detach().cpu().numpy() for name, value in obs["proprio_obs"].items()}
        assert all(np.isfinite(value).all() for value in state.values())
        assert not bool(terminated.any()) and not bool(truncated.any()), "Unexpected early termination"
        for name in ("banana", "bowl", "robot"):
            assert torch.isfinite(env.scene[name].data.root_state_w).all()
        state_trace.append({name: value.tolist() for name, value in state.items()})
        frames.append(obs["image_obs"]["wrist_cam"][0].detach().cpu().numpy())
        report["policy_actions_executed"] += 1
    np.save(OUT / "executed-action-rows.npy", mapped[:8])
    np.save(OUT / "post-step-wrist-frames.npy", np.stack(frames))
    cv2.imwrite(str(OUT / "wrist-after-eight-steps.png"), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    report["post_step_proprioception"] = state_trace
    report["simulated_seconds_executed"] = 8 * float(env.step_dt)
    report["all_post_step_states_finite"] = True
    report["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    (OUT / "scene-inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DISPOSABLE_RELOAD_EXECUTION_COMPLETE", json.dumps(report), flush=True)
except BaseException as exc:
    traceback.print_exc()
    (OUT / "smoke-error.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}, indent=2) + "\n")
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    if client is not None:
        client.close()
    if env is not None:
        env.close()
    app.close()
