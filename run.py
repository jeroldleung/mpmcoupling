import taichi as ti
import materialpoint as mp
import grid
import mpmsolver
import config

ti.init(arch=ti.vulkan)

jelly = mp.Jelly(rho=500, width=0.2)
water = mp.Water(rho=1000, width=0.4)
print(f'jelly: {jelly.num}')
print(f'water: {water.num}')

grid1 = grid.Grid()
grid2 = grid.Grid()


def init():
    config.init_material(jelly, center=ti.Vector([0.7, 0.5]), rotate=False)
    config.init_material(water, center=ti.Vector([0.2, 0.5]), rotate=False)


init()
window = ti.ui.Window('MPM Coupling',
                      res=(512, 512),
                      show_window=config.show_window)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))

frame = 1

while window.running:
    for _ in range(config.substeps):
        if ti.static(config.show_window):
            if window.get_event(ti.ui.PRESS):
                if window.event.key == 'r':
                    init()
        grid1.clear()
        grid2.clear()

        mpmsolver.particle_to_grid(jelly, grid1)
        mpmsolver.particle_to_grid(water, grid2)

        mpmsolver.update_grid_vel(grid1, grid2)

        grid1.boundary_handle(config.bound, False)
        grid2.boundary_handle(config.bound - 1, True)

        mpmsolver.grid_to_particle(grid1, jelly)
        mpmsolver.grid_to_particle(grid2, water)

    canvas.circles(jelly.px, radius=0.005, color=(1.0, 0.5, 0.8))
    canvas.circles(water.px, radius=0.005, color=(0.5, 0.8, 0.9))

    if ti.static(config.save_image):
        window.save_image(
            f"{frame:04d}.png"
        )  # You must run the program in your saved file directory
        if frame == 500: break
        frame += 1
    if ti.static(config.show_window):
        window.show()
