'''Module containing functions to do the boundary condition survey'''

import matplotlib.pyplot as plt
import numpy as np

# Set up our arrays
def set_arrays(npts):
    eta_now = np.zeros((npts, npts))
    u_now = np.zeros_like(eta_now)
    v_now = np.zeros_like(eta_now)
    eta_prev = np.zeros_like(eta_now)
    u_prev = np.zeros_like(eta_now)
    v_prev = np.zeros_like(eta_now)
    eta_next = np.zeros_like(eta_now)
    u_next = np.zeros_like(eta_now)
    v_next = np.zeros_like(eta_now)
    return eta_now, u_now, v_now, eta_prev, \
           u_prev, v_prev, eta_next, u_next, v_next


# Initialize Gaussian
def initialize(magnitude, gausshalfwidth, eta):
    eta_init = np.zeros_like(eta)
    n = eta.shape[0]
    half = (n - 1) / 3.
    islice = np.arange(n)
    jslice = np.arange(n)
    ijslice, jislice = np.meshgrid(islice, jslice)
    eta_init = magnitude * np.exp(-(
        (ijslice - half)**2 + (jislice - half)**2) 
                                  / (2 * gausshalfwidth**2))
    return eta_init


# A 2-dimensional Gaussian if you want it
def initialize_simple(magnitude, gausshalfwidth, eta):
    eta_init = np.zeros_like(eta)
    n = eta.shape[0]
    half = (n - 1) / 3.
    islice = np.arange(n)
    jslice = np.arange(n)
    ijslice, jislice = np.meshgrid(islice, jslice)
    eta_init = magnitude * np.exp(-(
        (jislice - half)**2) / (2 * gausshalfwidth**2))
    return eta_init

def incoming_wave(t, npts, dx, f, g, wavespeed):
    eta0 = 0.2
    wavenumber = 2 * np.pi / (10 * npts * dx)
    omega = wavespeed * wavenumber
    Xs = np.arange(npts) * dx
    Ys = np.arange(npts) * dx
    X, Y = np.meshgrid(Xs, Ys)
    eta_wave = eta0 * np.sin(wavenumber * Y - omega * t)
    u_wave = -g * eta0 * wavenumber * omega * np.sin(wavenumber * (
        Y + dx / 2) - omega * t) / (f * f - omega * omega)
    v_wave = g * eta0 * f * wavenumber * np.cos(wavenumber * Y - omega * t) / (
        f * f - omega * omega)
    return eta_wave, u_wave, v_wave

# Euler Step
def euler(eta, u, v, idt, f, g, H, dx):
    n = eta.shape[0]
    eta_next = np.zeros_like(eta)
    u_next = np.zeros_like(eta)
    v_next = np.zeros_like(eta)
    eta_next[1:, 1:] = eta[1:, 1:] + (idt * (-H) 
                                    * (u[1:, 1:] - u[:-1, 1:] +
                                       v[1:, 1:] - v[1:, :-1])) / dx
    u_next[:-1, 1:] = u[:-1, 1:] + idt * (
        -g * (eta[1:, 1:] - eta[:-1, 1:]) / dx + 0.25 * f *
        (v[:-1, 1:] + v[1:, 1:] + v[:-1, :-1] + v[1:, :-1]))
    v_next[1:, :-1] = v[1:, :-1] + idt * (
        -g * (eta[1:, 1:] - eta[1:, :-1]) / dx - 0.25 * f *
        (u[1:, :-1] + u[:-1, :-1] + u[1:, 1:] + u[:-1, 1:]))
    return eta_next, u_next, v_next

# Leap-frog Step
def leapfrog(eta, u, v, etap, up, vp, idt, f, g, H, dx):
    eta_next = np.zeros_like(eta)
    u_next = np.zeros_like(eta)
    v_next = np.zeros_like(eta)
    eta_next[1:, 1:] = etap[1:, 1:] + idt * (-H) * (
        u[1:, 1:] - u[:-1, 1:] + v[1:, 1:] - v[1:, :-1]) / dx
    u_next[:-1, 1:] = up[:-1, 1:] + idt * (
        -g * (eta[1:, 1:] - eta[:-1, 1:]) / dx + 0.25 * f *
        (v[:-1, 1:] + v[1:, 1:] + v[:-1, :-1] + v[1:, :-1]))
    v_next[1:, :-1] = vp[1:, :-1] + idt * (
        -g * (eta[1:, 1:] - eta[1:, :-1]) / dx - 0.25 * f *
        (u[1:, :-1] + u[:-1, :-1] + u[1:, 1:] + u[:-1, 1:]))
    return eta_next, u_next, v_next

def unstagger(ugrid, vgrid):
    """Interpolate u and v component values to values at grid cell centres.

    The shapes are the returned arrays are 1 less than those of
    the input arrays in the y and x dimensions.

    :arg ugrid: u velocity component values with axes (..., y, x)
    :type ugrid: :py:class:`numpy.ndarray`

    :arg vgrid: v velocity component values with axes (..., y, x)
    :type vgrid: :py:class:`numpy.ndarray`

    :returns u, v: u and v component values at grid cell centres
    :rtype: 2-tuple of :py:class:`numpy.ndarray`
        """
    u = np.add(ugrid[..., :-1], ugrid[..., 1:]) / 2
    v = np.add(vgrid[..., :-1, :], vgrid[..., 1:, :]) / 2
    return u[..., 1:, :], v[..., 1:]

def make_plot(npts, dx, eta_next, u_next, v_next):
    '''make a contour plot of surface height
    and a quiver plot of velocity'''
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 7))

    Xs = np.arange(npts)*dx
    Ys = np.arange(npts)*dx
    X, Y = np.meshgrid(Xs, Ys)
    mesh = ax.pcolormesh(X[1:, 1:], Y[1:, 1:], 
                     np.transpose(eta_next[1:, 1:]), cmap='plasma')
    fig.colorbar(mesh, ax=ax, orientation='horizontal')
    ax.set_xlim((dx, 31*dx))
    ax.set_ylim((dx, 31*dx))
    us, vs = unstagger(u_next, v_next)
    ax.quiver(X[1::3, 1::3], Y[1::3, 1::3], np.transpose(us[::3, ::3]), np.transpose(vs[::3, ::3]),
               pivot='mid')

def make_contourplots(npts, dx, eta_next, u_next, v_next):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.8))
    mesh = axs[0].pcolormesh(np.transpose(eta_next[1:, 1:]), cmap='plasma')
    fig.colorbar(mesh, ax=axs[0], orientation='horizontal')
    axs[0].set_xlim((0,31))
    axs[0].set_ylim((0,31))
    mesh = axs[1].pcolormesh(np.transpose(u_next[:, 1:]), cmap=cm.curl)
    fig.colorbar(mesh, ax=axs[1], orientation='horizontal')
    axs[1].set_xlim((0,32))
    axs[1].set_ylim((0,31))
    mesh = axs[2].pcolormesh(np.transpose(v_next[1:]), cmap=cm.curl)
    fig.colorbar(mesh, ax=axs[2], orientation='horizontal')
    axs[2].set_xlim((0,31))
    axs[2].set_ylim((0,32));

def make_line_plots(eta_next, u_next, v_next, direction, ii):
    '''make line plots of surface height and velocity either in
    eastwest or northsouth direction at grid point ii'''
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.8))
    if direction == 'eastwest':
        axs[0].plot(eta_next[1:, ii], 'bo-')
        axs[1].plot(u_next[:, ii], 'ro-')
        axs[2].plot(v_next[1:, ii], 'go-')
    elif direction == 'northsouth':
        axs[0].plot(eta_next[ii, 1:], 'bo-')
        axs[1].plot(u_next[ii, :], 'ro-')
        axs[2].plot(v_next[ii, 1:], 'go-')
    else:
        print ('Direction unknown, use eastwest or northsouth')
    axs[0].set_title('Surface Height')
    axs[1].set_title('East Velocity')
    axs[2].set_title('North Velocity')
