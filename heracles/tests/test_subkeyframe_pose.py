import numpy as np

from heracles.pose_math import quat_mul, quat_rotate


def test_quat_rotate_identity():
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(quat_rotate((1.0, 0.0, 0.0, 0.0), v), v)


def test_quat_rotate_90deg_z():
    # +90 deg about Z maps x-axis -> y-axis
    q = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    assert np.allclose(quat_rotate(q, np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-9)


def test_quat_mul_identity():
    q = (0.5, 0.5, 0.5, 0.5)
    assert np.allclose(quat_mul((1.0, 0.0, 0.0, 0.0), q), q)


def test_compose_world_pose():
    from heracles.pose_math import compose_pose
    # anchor at (1,0,0), identity rotation; relative +0.5 x
    world_t, world_R = compose_pose(
        world_t_anchor=np.array([1.0, 0.0, 0.0]),
        world_R_anchor=(1.0, 0.0, 0.0, 0.0),
        anchor_t_sub=np.array([0.5, 0.0, 0.0]),
        anchor_R_sub=(1.0, 0.0, 0.0, 0.0),
    )
    assert np.allclose(world_t, [1.5, 0.0, 0.0])
    assert np.allclose(world_R, [1.0, 0.0, 0.0, 0.0])
