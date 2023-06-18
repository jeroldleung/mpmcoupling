import taichi as ti
import materialpoint
import grid
import tools
import config


@ti.kernel
def particle_to_grid(material: ti.template(), grid: ti.template()):
    for p in range(material.num):
        Xp = material.px[p] * config.inv_dx
        base = int(Xp - 0.5)
        stress = material.compute_stress(p)
        affine = -config.dt * material.volume * 4 * config.inv_dx**2 * stress + material.mass * material.C[
            p]
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * config.dim)))):
            I = base + offset
            w = tools.weight(Xp - I)
            grid.velocity[I] += w * (material.mass * material.pv[p] +
                                     affine @ (I * config.dx - material.px[p]))
            grid.mass[I] += material.mass * w
            grid.normal[I] += material.mass * w * (I - Xp)


@ti.kernel
def update_grid_vel(grid1: ti.template(), grid2: ti.template()):
    for I in ti.grouped(ti.ndrange(*((config.n_grids, ) * config.dim))):
        grid1.update_grid_vel(I)
        grid2.update_grid_vel(I)
        m1, m2 = grid1.mass[I], grid2.mass[I]
        if m1 > 0.0 and m2 > 0.0:
            v1, v2 = grid1.velocity[I], grid2.velocity[I]
            ni = (grid1.normal[I] -
                  grid2.normal[I]).normalized()  # grid normal
            vir = v1 - v2  # grid responsed velocity
            if vir.dot(ni) > 0.0:
                v1n, v2n = v1.dot(ni) * ni, v2.dot(ni) * ni
                v1t, v2t = v1 - v1n, v2 - v2n

                slide = ti.max((m2 - m1) / (m1 + m2),
                               0.0)  # adaptive sliding fractor
                # slide = 1.0 # user-controlable sliding fractor
                vi = (m1 * v1 + m2 * v2) / (m1 + m2)
                v1t_new = slide * m2 * (v1t - v2t) / (m1 + m2)
                v2t_new = slide * m1 * (v2t - v1t) / (m1 + m2)
                grid1.velocity[I] = vi + v1t_new
                grid2.velocity[I] = vi + v2t_new


@ti.kernel
def grid_to_particle(grid: ti.template(), material: ti.template()):
    for p in range(material.num):
        Xp = material.px[p] * config.inv_dx
        base = int(Xp - 0.5)
        v_pic = ti.Vector.zero(float, config.dim)
        new_C = ti.Matrix.zero(float, config.dim, config.dim)
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * config.dim)))):
            I = base + offset
            w = tools.weight(Xp - I)
            v_pic += grid.velocity[I] * w
            new_C += w * grid.velocity[I].outer_product(I * config.dx -
                                                        material.px[p])
        if not material.pinned:
            material.pv[p] = v_pic
        material.C[p] = 4 * config.inv_dx * config.inv_dx * new_C
        material.deformation_gradient_update(p)
        material.particle_advect(p)
