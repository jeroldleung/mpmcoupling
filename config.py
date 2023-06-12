import taichi as ti

dim = 2
n_grids = 100
dx = 1 / n_grids
inv_dx = 1 / dx
dt = 2e-4
substeps = 20
bound = 10

show_window = True
save_image = False


@ti.func
def circle2d(r):
    rn = ti.sqrt(r * ti.random())
    thetan = 2 * 3.1415926 * ti.random()
    return rn * ti.Vector([ti.cos(thetan), ti.sin(thetan)])


@ti.kernel
def init_material(material: ti.template(), center: ti.template(),
                  rotate: ti.template()):
    for p in range(material.num):
        # material.px[p] = circle2d(material.width) * material.width + center
        material.px[p] = ti.Vector([ti.random() for i in range(dim)
                                    ]) * material.width + center
        material.pv[p] = ti.Vector.zero(float, dim)
        material.C[p] = ti.Matrix.zero(float, dim, dim)
        material.F[p] = ti.Matrix.identity(float, dim)
        material.J[p] = 1

        if ti.static(rotate):
            rotation = ti.Matrix.identity(float, dim)
            if ti.static(dim == 3):
                rotation = ti.Matrix([[1, 0, 0],
                                      [0, ti.cos(0.785), -ti.sin(0.785)],
                                      [0, ti.sin(0.785),
                                       ti.cos(0.785)]])  # in radians
            else:
                rotation = ti.Matrix([[ti.cos(0.785), -ti.sin(0.785)],
                                      [ti.sin(0.785),
                                       ti.cos(0.785)]])  # in radians

            offset = center + ti.Vector(
                [material.width * 0.5 for i in range(dim)])
            material.px[p] -= offset
            material.px[p] = rotation @ material.px[p]
            material.px[p] += offset
