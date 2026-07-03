"""Minimal quaternion pose math (w, x, y, z convention), no scipy dependency."""
import numpy as np


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q, v):
    w, x, y, z = q
    vq = (0.0, v[0], v[1], v[2])
    q_conj = (w, -x, -y, -z)
    rotated = quat_mul(quat_mul(q, vq), q_conj)
    return rotated[1:]


def compose_pose(world_t_anchor, world_R_anchor, anchor_t_sub, anchor_R_sub):
    """world_T_sub = world_T_anchor * anchor_T_sub. Returns (t (3,), R (w,x,y,z))."""
    world_t_sub = np.asarray(world_t_anchor) + quat_rotate(world_R_anchor, np.asarray(anchor_t_sub))
    world_R_sub = quat_mul(world_R_anchor, anchor_R_sub)
    return world_t_sub, world_R_sub
