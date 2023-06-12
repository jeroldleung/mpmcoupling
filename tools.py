import taichi as ti
import numpy as np
import config


# one-dimensional
@ti.func
def quadratic_spline(distance):
    dis = abs(distance)
    dis2 = dis**2
    res = 0.0
    if dis < 1.5:
        if dis < 0.5:
            res = 0.75 - dis2
        else:
            res = 0.5 * (1.5 - dis)**2
    return res


@ti.func
def weight(r):
    w = 1.0
    for i in ti.static(range(r.n)):  # n is number of rows
        w *= quadratic_spline(r[i])
    return w


def save_ply(frame, px, n_particles, path):
    pos = px.to_numpy()
    writer = ti.tools.PLYWriter(num_vertices=n_particles)
    writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
    writer.export_frame_ascii(frame, path)
