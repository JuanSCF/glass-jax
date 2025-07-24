# import numpy as np
# import healpy as hp
# import matplotlib.pyplot as plt
# import glass.jax as jglass
# import jax_cosmo as jc
# import jax 

# # use the CAMB cosmology that generated the matter power spectra
# import camb
# from cosmology import Cosmology

# # GLASS modules: cosmology and everything in the glass namespace
# import glass.shells
# import glass.fields
# import glass.shapes
# import glass.lensing
# import glass.observations

# import glass._src.camb
# # cosmology for the simulation
# h = 0.7
# Oc = 0.25
# Ob = 0.05

# # basic parameters of the simulation
# nside = lmax = 256

# # set up CAMB parameters for matter angular power spectrum
# pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2,
#                        NonLinear=camb.model.NonLinear_both)

# # get the cosmology from CAMB
# cosmo = Cosmology.from_camb(pars)
# cosmo = jc.Planck15() 

# # %%
# # Set up the matter sector.

# # shells of 200 Mpc in comoving distance spacing
# zb = glass.shells.distance_grid(cosmo, 0., 3., dx=200.)

# # tophat window function for shells
# zs, ws = glass.shells.tophat_windows(zb)

# # compute the angular matter power spectra of the shells with CAMB
# cls = glass.camb.matter_cls(pars, lmax, zs, ws)
# np.save("cls.npy", cls)
# cls = np.load("cls.npy", allow_pickle=True)
# # compute Gaussian cls for lognormal fields for 3 correlated shells
# # putting nside here means that the HEALPix pixel window function is applied
# # gls = glass.fields.lognormal_gls(cls, nside=nside, lmax=lmax, ncorr=3)
# # gls = glass.fields.generate_lognormal(cls, nside=nside)

# # generator for lognormal matter fields

# import numpy as np
# from scipy.special import legendre
# from scipy.interpolate import interp1d

# class C2wFunc:
#     def __init__(self, theta_array):
#         self.theta_array = theta_array

#     def __call__(self, C_ell):
#         # compute w(theta) for input C_ell
#         w_theta = np.zeros_like(self.theta_array)
#         ell = np.arange(len(C_ell))
#         for i, theta in enumerate(self.theta_array):
#             P_ell = legendre(ell)(np.cos(theta))
#             w_theta[i] = np.sum((2*ell+1)/(4*np.pi)*C_ell*P_ell)
#         return self.theta_array, w_theta

# class W2CFunc:
#     def __init__(self, ell_array, theta_array):
#         self.ell_array = ell_array
#         self.theta_array = theta_array

#     def __call__(self, w_theta):
#         from scipy.integrate import simps
#         C_ell = np.zeros_like(self.ell_array)
#         for i, l in enumerate(self.ell_array):
#             P_l = legendre(l)(np.cos(self.theta_array))
#             integrand = w_theta * P_l * np.sin(self.theta_array)
#             C_ell[i] = 2 * np.pi * simps(integrand, self.theta_array)
#         return self.ell_array, C_ell
# theta = np.linspace(0, np.pi, 100)  # angular scales
# ell = np.arange(lmax+1)

# C2w_f = C2wFunc(theta)
# w2C_f = W2CFunc(ell, theta)

# gls=glass.fields.generate_lognormal(cls, shifts, C2w_f, w2C_f, nside, L, seed)

# number_of_shells = len(gls)
# # shifts values????
# shifts = [1.0 for _ in range(number_of_shells)]  # simple default

# matter = glass.fields.generate_lognormal(
#     gls=gls, shifts=shifts, C2w_f=C2w_f, w2C_f=w2C_f, nside=nside, L=lmax, seed=0)
# # np.save('cls.npy', cls)
# # %%
# # Set up the lensing sector.

# # this will compute the convergence field iteratively
# convergence = glass.lensing.MultiPlaneConvergence(cosmo)

# # %%
# # Set up the galaxies sector.

# # galaxy density (using 1/100 of the expected galaxy number density for Stage-IV)
# n_arcmin2 = 0.3

# # true redshift distribution following a Smail distribution
# z = np.arange(0.01, 3., 0.01)
# dndz = glass.observations.smail_nz(z, z_mode=0.9, alpha=2., beta=1.5)
# dndz *= n_arcmin2

# # compute bin edges with equal density
# nbins = 5
# zedges = glass.observations.equal_dens_zbins(z, dndz, nbins=nbins)

# # photometric redshift error
# sigma_z0 = 0.03

# # split distribution by tomographic bin, assuming photometric redshift errors
# tomo_nz = glass.observations.tomo_nz_gausserr(z, dndz, sigma_z0, zedges)

# # constant bias parameter for all shells
# bias = 1.2

# # ellipticity standard deviation as expected for a Stage-IV survey
# sigma_e = 0.27

# for i, delta_i in enumerate(matter):
        
#     ia = jglass.intrinsic_alignments.get_IA(zb[i], delta_i, nside, A1=0.18, bTA=0.8, A2=0.1, model='NLA')

#     hp.mollview(np.log10(2+delta_i))
#     plt.show()

#     hp.mollview(ia.real)
#     plt.show()

#     hp.mollview(ia.imag)
#     plt.show()


#     ia = jglass.intrinsic_alignments.get_IA(zb[i], delta_i, nside, A1=0.18, bTA=0.8, A2=0.1, model='TATT')

#     hp.mollview(ia.real)
#     plt.show()

#     hp.mollview(ia.imag)
#     plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 0) Imports & Monkey‑patch jnp.trapz (needed by smail_nz)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import camb
import jax_cosmo as jc
from scipy.special import legendre
from scipy.integrate import simpson

# Monkey‑patch trapz into jax.numpy so glass.observations works
def _trapz(a, x=None, axis=-1):
    if x is None:
        dx = 1.0
    else:
        dx = jnp.diff(x, axis=0)
    # pairwise trapezoids:
    a0 = jnp.take(a, jnp.arange(a.shape[axis]-1), axis=axis)
    a1 = jnp.take(a, jnp.arange(1, a.shape[axis]), axis=axis)
    trape = (a0 + a1) / 2 * dx
    return jnp.sum(trape, axis=axis)

jnp.trapz = _trapz


# ─────────────────────────────────────────────────────────────────────────────
# 1) Local modules & cosmology setup
# ─────────────────────────────────────────────────────────────────────────────
from cosmology import Cosmology      # your own wrapper
import glass.shells
import glass.fields
import glass.lensing
import glass.observations
import glass.jax as jglass

# Simulation parameters
h = 0.7; Oc = 0.25; Ob = 0.05
nside = 32
lmax  = 2 * nside   # = 512, safely ≥ 2*nside
# lmax = 3 * nside - 1

# CAMB parameters
pars = camb.set_params(
    H0=100*h,
    omch2=Oc*h**2,
    ombh2=Ob*h**2,
    NonLinear=camb.model.NonLinear_both
)

# Use jax_cosmo Planck15 for distances
cosmo_jc = jc.Planck15()


# ─────────────────────────────────────────────────────────────────────────────
# 2) Compute CAMB Cls for tophat shells
# ─────────────────────────────────────────────────────────────────────────────
# comoving-distance shells (200 Mpc spacing)
zb = glass.shells.distance_grid(cosmo_jc, 0.0, 3.0, dx=200.0)
# tophat redshift windows
zs, ws = glass.shells.tophat_windows(zb)
# matter power spectra
cls = glass._src.camb.matter_cls(pars, lmax, zs, ws)

# ensure each entry is at least 2D so cl[0] is always an array
cls = [np.atleast_2d(c) for c in cls]

# debug: print shapes
# for i, c in enumerate(cls):
    # print(f"cls[{i}].shape = {c.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 3) Build C2w_f and w2C_f helpers
# ─────────────────────────────────────────────────────────────────────────────
from scipy.special import eval_legendre

class C2wFunc:
    def __init__(self, theta_array):
        self.theta_array = theta_array
        self.x = theta_array

    def __call__(self, C_ell):
        w_theta = np.zeros_like(self.theta_array)
        ell = np.arange(len(C_ell))
        for i, theta in enumerate(self.theta_array):
            # vectorized Legendre evaluation
            P_ell = eval_legendre(ell, np.cos(theta))
            w_theta[i] = np.sum((2*ell+1)/(4*np.pi) * C_ell * P_ell)
        return self.theta_array, w_theta

class W2CFunc:
    def __init__(self, ell_array, theta_array):
        self.ell_array = ell_array
        self.theta_array = theta_array

    def __call__(self, w_theta):
        C_ell = np.zeros_like(self.ell_array)
        for i, l in enumerate(self.ell_array):
            # here l is scalar, so eval_legendre works too
            P_l = eval_legendre(l, np.cos(self.theta_array))
            integrand = w_theta * P_l * np.sin(self.theta_array)
            C_ell[i] = 2*np.pi * simpson(integrand, self.theta_array)
        return self.ell_array, C_ell

theta = np.linspace(1e-3, np.pi-1e-3, 200)
ell   = np.arange(lmax+1)
C2w_f = C2wFunc(theta)
w2C_f = W2CFunc(ell, theta)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Generate lognormal spectra & maps
# ─────────────────────────────────────────────────────────────────────────────
n_shells = len(cls)
# shifts   = [1.0] * n_shells

shifts = []
for cl in cls:
    cl = cl[0]  # shape is (1, lmax+1)
    # Estimate variance in real space via sum over C_ell:
    ell = np.arange(len(cl))
    var = np.sum((2 * ell + 1) * cl / (4 * np.pi))
    shifts.append(var)
shifts = np.array(shifts)

seed     = jax.random.PRNGKey(42)

gls = glass.fields.generate_lognormal(
    cls, shifts, C2w_f, w2C_f, nside, lmax, seed
)
matter_maps = list(gls)   # evaluate generator
print("Generated maps:", len(matter_maps))


# ─────────────────────────────────────────────────────────────────────────────
# 5) Lensing & galaxy settings
# ─────────────────────────────────────────────────────────────────────────────
# positional argument only (no keyword)
convergence = glass.lensing.MultiPlaneConvergence(cosmo_jc)

n_arcmin2 = 0.3
zgrid     = np.arange(0.01, 3.0, 0.01)
dndz      = glass.observations.smail_nz(zgrid, z_mode=0.9, alpha=2.0, beta=1.5)
dndz     *= n_arcmin2
nbins     = 5
zedges    = glass.observations.equal_dens_zbins(zgrid, dndz, nbins=nbins)
sigma_z0  = 0.03
tomo_nz   = glass.observations.tomo_nz_gausserr(zgrid, dndz, sigma_z0, zedges)
bias      = 1.2
sigma_e   = 0.27


# ─────────────────────────────────────────────────────────────────────────────
# 6) Plot matter + IA per shell
# ─────────────────────────────────────────────────────────────────────────────
for i, delta_i in enumerate(matter_maps):
    ia_nla = jglass.intrinsic_alignments.get_IA(
        zb[i], delta_i, nside,
        A1=0.18, bTA=0.8, A2=0.1, model='NLA'
    )
    ia_tatt = jglass.intrinsic_alignments.get_IA(
        zb[i], delta_i, nside,
        A1=0.18, bTA=0.8, A2=0.1, model='TATT'
    )

    hp.mollview(np.log10(2 + delta_i), title=f"Matter Shell {i}")
    plt.show()

    hp.mollview(ia_nla.real, title=f"IA NLA Shell {i} (real)")
    plt.show()
    hp.mollview(ia_nla.imag, title=f"IA NLA Shell {i} (imag)")
    plt.show()

    hp.mollview(ia_tatt.real, title=f"IA TATT Shell {i} (real)")
    plt.show()
    hp.mollview(ia_tatt.imag, title=f"IA TATT Shell {i} (imag)")
    plt.show()
