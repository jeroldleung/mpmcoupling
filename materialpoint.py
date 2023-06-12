import taichi as ti
import config


@ti.data_oriented
class MaterialPoint:

    def __init__(self, rho, width):
        self.volume = (config.dx * 0.5)**config.dim
        self.width = width
        self.mass = rho * self.volume
        self.num = int(2 * width**config.dim / self.volume)
        self.px = ti.Vector.field(config.dim, float, self.num)
        self.pv = ti.Vector.field(config.dim, float, self.num)
        self.C = ti.Matrix.field(config.dim, config.dim, float, self.num)
        self.F = ti.Matrix.field(config.dim, config.dim, float, self.num)
        self.J = ti.field(float, self.num)
        self.pinned = False

    @ti.func
    def particle_advect(self, p):
        self.px[p] += config.dt * self.pv[p]

    def pin(self):
        self.pinned = True

    def unpin(self):
        self.pinned = False


@ti.data_oriented
class Water(MaterialPoint):

    def __init__(self, rho, width):
        super(Water, self).__init__(rho, width)
        self.K = 1e5  # bulk modulus of water

    @ti.func
    def compute_stress(self, p):
        Jp = self.J[p]
        pressure = self.K * (1 - Jp)
        stress = -pressure * ti.Matrix.identity(float, config.dim)
        return stress

    @ti.func
    def deformation_gradient_update(self, p):
        self.J[p] = (1.0 + config.dt * self.C[p].trace()) * self.J[p]


@ti.data_oriented
class Jelly(MaterialPoint):

    def __init__(self, rho, width):
        super(Jelly, self).__init__(rho, width)
        self.E, self.nu = 3e4, 0.2  # Young's modulus and Poisson's ratio
        self.mu, self.la = self.E / (2 * (1 + self.nu)), self.E * self.nu / (
            (1 + self.nu) * (1 - 2 * self.nu))  # Lame parameters

    @ti.func
    def compute_stress(self, p):
        Jp = ti.Matrix.determinant(self.F[p])
        U, sig, V = ti.svd(self.F[p])
        stress = 2 * self.mu * (self.F[p] - U @ V.transpose()) @ self.F[
            p].transpose() + self.la * (Jp - 1) * Jp * ti.Matrix.identity(
                float, config.dim)  # Fixed Corotated
        return stress

    @ti.func
    def deformation_gradient_update(self, p):
        self.F[p] = (ti.Matrix.identity(float, config.dim) +
                     config.dt * self.C[p]) @ self.F[p]


@ti.data_oriented
class Sand(MaterialPoint):

    def __init__(self, rho, width):
        super(Sand, self).__init__(rho, width)
        self.ap = ti.field(float, self.num)
        self.qp = ti.field(float, self.num)
        self.h0, self.h1, self.h2, self.h3 = 35, 0, 0.2, 10
        self.pi = 3.14159265358979
        self.vcs = ti.field(float, self.num)  # prevent volume gain

        self.E, self.nu = 3e5, 0.2  # Young's modulus and Poisson's ratio
        self.mu, self.la = self.E / (2 * (1 + self.nu)), self.E * self.nu / (
            (1 + self.nu) * (1 - 2 * self.nu))  # Lame parameters

    @ti.func
    def compute_stress(self, p):
        U, sig, V = ti.svd(self.F[p])
        inv_sig = sig.inverse()
        e = ti.Matrix.identity(float, config.dim)
        for i in ti.static(range(config.dim)):
            e[i, i] = ti.log(sig[i, i])
        PK1 = U @ (2 * self.mu * inv_sig @ e + self.la * e.trace() *
                   inv_sig) @ V.transpose()  # Drucker-Prager
        stress = PK1 @ self.F[p].transpose()
        return stress

    @ti.func
    def project(self, sig, p):
        e0 = ti.Matrix.identity(float, config.dim)
        for i in ti.static(range(config.dim)):
            e0[i, i] = ti.log(sig[i, i])
        e = e0 + self.vcs[p] / config.dim * ti.Matrix.identity(
            float, config.dim)  # prevent volume gain
        ehat = e - e.trace() / config.dim * ti.Matrix.identity(
            float, config.dim)
        ehat_sqsum = 0.0
        for i in ti.static(range(config.dim)):
            ehat_sqsum += ehat[i, i]**2
        Fnorm = ti.sqrt(ehat_sqsum)  # Frobenius norm
        yp = Fnorm + (config.dim * self.la +
                      2 * self.mu) / (2 * self.mu) * e.trace() * self.ap[p]
        new_sig = ti.Matrix.zero(float, config.dim, config.dim)
        dq = 0.0
        if yp <= 0.0:  # Case I
            new_sig = sig
            dq = 0.0
        elif Fnorm == 0 or e.trace() > 0:  # Case II
            new_sig = ti.Matrix.identity(float, config.dim)
            e_sqsum = 0.0
            for i in ti.static(range(config.dim)):
                e_sqsum += e[i, i]**2
            dq = ti.sqrt(e_sqsum)
        else:
            Hp = e - yp / Fnorm * ehat
            new_sig = ti.Matrix.identity(float, config.dim)
            for i in ti.static(range(config.dim)):
                new_sig[i, i] = ti.exp(Hp[i, i])
            dq = yp
        return new_sig, dq

    @ti.func
    def hardening(self, dq, p):
        self.qp[p] += dq
        # phi = self.h0 + (self.h1 * self.qp[p] - self.h3) * ti.exp(-self.h2 * self.qp[p])
        phi = 40.0
        phi = phi / 180 * self.pi  # details in Table. 3: Friction angle phi_F and hardening parameters h0, h1, and h3 are listed in degrees for convenience
        sin_phi = ti.sin(phi)
        self.ap[p] = ti.sqrt(2 / 3) * (2 * sin_phi) / (3 - sin_phi)

    @ti.func
    def deformation_gradient_update(self, p):
        self.F[p] = (ti.Matrix.identity(float, config.dim) +
                     config.dt * self.C[p]) @ self.F[p]
        U, sig, V = ti.svd(self.F[p])
        new_sig, dq = self.project(sig, p)
        self.hardening(dq, p)
        new_F = U @ new_sig @ V.transpose()
        self.vcs[p] += ti.log(self.F[p].determinant()) - ti.log(
            new_F.determinant())
        self.F[p] = new_F


@ti.data_oriented
class Snow(MaterialPoint):

    def __init__(self, rho, width):
        super(Snow, self).__init__(rho, width)
        self.E, self.nu = 1e5, 0.2  # Young's modulus and Poisson's ratio
        self.mu, self.la = self.E / (2 * (1 + self.nu)), self.E * self.nu / (
            (1 + self.nu) * (1 - 2 * self.nu))  # Lame parameters

    @ti.func
    def compute_stress(self, p):
        h = ti.exp(10 * (1.0 - self.J[p]))
        mu, la = self.mu * h, self.la * h
        U, sig, V = ti.svd(self.F[p])
        J = 1.0
        for i in ti.static(range(config.dim)):
            new_sig = ti.min(ti.max(sig[i, i], 1 - 2.5e-2),
                             1 + 4.5e-3)  # Plasticity
            self.J[p] *= sig[i, i] / new_sig
            sig[i, i] = new_sig
            J *= new_sig
        self.F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (self.F[p] -
                           U @ V.transpose()) @ self.F[p].transpose() + la * (
                               J - 1) * J * ti.Matrix.identity(
                                   float, config.dim)  # Fixed Corotated
        return stress

    @ti.func
    def deformation_gradient_update(self, p):
        self.F[p] = (ti.Matrix.identity(float, config.dim) +
                     config.dt * self.C[p]) @ self.F[p]
