import taichi as ti
import config


@ti.data_oriented
class Grid:

    def __init__(self):
        self.mass = ti.field(float, (config.n_grids, ) * config.dim)
        self.velocity = ti.Vector.field(config.dim, float,
                                        (config.n_grids, ) * config.dim)
        self.normal = ti.Vector.field(config.dim, float,
                                      (config.n_grids, ) * config.dim)

    @ti.func
    def update_grid_vel(self, I):
        if self.mass[I] > 0:
            self.velocity[I] /= self.mass[I]
            self.normal[I] = self.normal[I].normalized()
            self.velocity[I][1] -= config.dt * 9.8

    @ti.kernel
    def boundary_handle(self, bound: ti.template(),
                        APPLY_FRICTION: ti.template()):
        for I in ti.grouped(ti.ndrange(*((config.n_grids, ) * config.dim))):
            cond = (I < bound) & (self.velocity[I] < 0) | (
                I > config.n_grids - bound) & (self.velocity[I] > 0)
            self.velocity[I] = ti.select(cond, 0, self.velocity[I])
            # apply friction
            if APPLY_FRICTION:
                mu_b = 0.75
                boundary_normal = ti.Vector.zero(float, config.dim)
                for d in ti.static(range(config.dim)):
                    index = I[d]
                    if index == bound and self.velocity[I][d] < 0.0:
                        boundary_normal[d] = 1.0
                    if index == config.n_grids - bound and self.velocity[I][
                            d] > 0.0:
                        boundary_normal[d] = -1.0
                if boundary_normal.any(
                ):  # Test whether any element not equal zero
                    vn = self.velocity[I].dot(boundary_normal)
                    if vn < 0:
                        v_tangent = self.velocity[I] - vn * boundary_normal
                        vt = v_tangent.norm()
                        if vt > 1e-12:
                            self.velocity[I] = v_tangent - (
                                vt if vt < -mu_b * vn else -mu_b *
                                vn) * v_tangent.normalized()

    @ti.kernel
    def clear(self):
        self.mass.fill(0)
        self.velocity.fill(ti.Vector.zero(float, config.dim))
        self.normal.fill(ti.Vector.zero(float, config.dim))
