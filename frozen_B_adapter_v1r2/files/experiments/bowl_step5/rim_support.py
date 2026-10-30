"""Read-only support/footprint proxy, with explicit force orientation."""
import numpy as np

VERTICAL_FORCE_N = .1
RIM_HEIGHT_TOLERANCE_M = .005
OVERLAP_TOLERANCE_M = 1e-6


def hull(points):
    points = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and cross(out[-2], out[-1], p) <= 0: out.pop()
            out.append(p)
        return out
    return np.asarray(half(points)[:-1] + half(points[::-1])[:-1])


def footprints_overlap(a, b):
    a, b = hull(a), hull(b)
    if len(a) < 3 or len(b) < 3: return False
    for polygon in (a, b):
        for edge in np.roll(polygon, -1, axis=0) - polygon:
            axis = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
            pa, pb = a @ axis, b @ axis
            if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= OVERLAP_TOLERANCE_M:
                return False
    return True


def support_signals(banana_box, bowl_box, sensor_forces, reversed_sensor=False):
    banana, bowl, forces = map(lambda a: np.asarray(a, dtype=float),
                              (banana_box, bowl_box, sensor_forces))
    assert banana.shape == bowl.shape == (8, 3)
    assert forces.size >= 3 and forces.shape[-1] == 3
    finite = all(np.isfinite(v).all() for v in (banana, bowl, forces))
    force = forces.reshape(-1, 3).sum(0) * (-1 if reversed_sensor else 1)
    lateral = np.linalg.norm(force[:2])
    supported = bool(finite and force[2] > VERTICAL_FORCE_N and force[2] >= lateral)
    overlap = bool(finite and footprints_overlap(banana[:, :2], bowl[:, :2]))
    reaches_rim = bool(finite and banana[:, 2].max() >= bowl[:, 2].max()-RIM_HEIGHT_TOLERANCE_M)
    return {'bowl_support': supported, 'rim_support': supported and overlap and reaches_rim,
            'support_force_on_banana_w': force.tolist(), 'support_sensor_reversed': bool(reversed_sensor),
            'footprints_overlap': overlap, 'object_reaches_rim_height': reaches_rim}
