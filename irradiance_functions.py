from __future__ import division, print_function, absolute_import
import scipy as sp
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


def read_irradiance(CodeLocation, AliasLocation):
    
    # ### LOAD / CALCULATE IRRADIANCE ###
    # Define the folder name of chosen location
    FolderNameIrradiance = os.path.join(os.getcwd(), f'Irradiance\\Spectra_{CodeLocation}_{AliasLocation}')

    # Simulate the irradiance data
    if not os.path.isdir(FolderNameIrradiance):
        # Irradiance(CodeLocation, AliasLocation)
        pass

    # Load the irradiance data, if it has been calculated already
    if 'irradiance' not in locals():
        irradiance = sp.io.loadmat(f'Irradiance/Spectra_{CodeLocation}_{AliasLocation}/Irr_spectra_clouds.mat')
        Data_TMY3 = sp.io.loadmat(f'Irradiance/Spectra_{CodeLocation}_{AliasLocation}/TMY3_{CodeLocation}_{AliasLocation}.mat')
        irradiance['Data_TMY3'] = Data_TMY3
        del Data_TMY3
        IndRefr = {}
        
    return irradiance

def wrap_to_180(angles):
    return (angles + 180) % 360 - 180


def wrap_to_360(angles):
    return angles % 360


def rotatesunangle(alpha, beta, gamma, phisun, thetasun):
    # first we map the angles to more feasible (mathematical) definitions
    phisun = wrap_to_180(np.array(phisun))
    thetasun = 90 - np.array(thetasun)

    # convert to radians & to angle space used by transformation functions
    phisun = np.deg2rad(phisun)
    thetasun = np.deg2rad(thetasun)

    # define euler matrix for rotation: 'ZYX'
    eul = np.deg2rad([alpha, beta, gamma])
    rotm = R.from_euler('ZYX', eul).as_quat()

    # transform sun coordinates to cartesian
    sx, sy, sz = np.sin(thetasun) * np.cos(phisun), np.sin(thetasun) * np.sin(phisun), np.cos(thetasun)

    # rotate vectors pointing to sun into rotated solar cell system
    # new coordinates describe sun viewed from rotated local coordinate system
    s = np.stack((sx, sy, sz), axis=-1)
    coord_rot = R.from_quat(rotm).apply(s)
    # transform back to spherical coordinates
    phisun_rot, thetasun_rot = np.arctan2(coord_rot[:, 1], coord_rot[:, 0]), np.arccos(
        coord_rot[:, 2] / np.linalg.norm(coord_rot, axis=1))

    # last transform back to degrees and initial angle definitions
    thetasun_rot = np.rad2deg(thetasun_rot)
    thetasun_rot = 90 - thetasun_rot
    phisun_rot = np.rad2deg(phisun_rot)
    phisun_rot = np.round(wrap_to_360(phisun_rot), 5)

    return phisun_rot, thetasun_rot


def trim_irradiance(lambda_range, I, w):
    startindex = np.where(w == lambda_range[0])[0][0]
    stopindex = np.where(w == lambda_range[-1])[0][0]
    d = lambda_range[1] - lambda_range[0]

    # Convert the step size to an integer index step
    step = int(d / (w[1] - w[0]))

    I_trimmed = I[:, startindex:stopindex + 1:step]

    return I_trimmed


def sph2cart(phi, theta, r):
    """Convert spherical coordinates to cartesian coordinates"""
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def get_illumination(alpha, beta, gamma):
    # normal of flat and unrotated solar cell pointing to zenith
    n = np.array([0, 0, 1])

    # define euler matrix for rotation 'ZYX'
    eul = np.deg2rad([alpha, -beta, gamma])
    rotm = R.from_euler('ZYX', eul).as_quat()

    # rotate normal about alpha and beta
    coord_rot = R.from_quat(rotm).apply(n)
    n_new = np.array([coord_rot[0], coord_rot[1], coord_rot[2]])

    # define local coordinates
    phi = np.linspace(-np.pi, np.pi, 361)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 181)

    # make meshgrid with phi and theta
    P, T = np.meshgrid(phi, theta)
    R_grid = np.ones_like(P)

    # transform the angle grid to cartesian coordinates
    px, py, pz = sph2cart(P, T, R_grid)
    p = np.stack((px, py, pz), axis=-1)

    # reshape the rotated normal to fit the size of the defined grid p
    n_new = np.tile(n_new.reshape(1, 1, 3), (p.shape[0], p.shape[1], 1))

    # calculate the angle between the rotated normal and the unrotated reference system p, both in cartesian coordinates
    A = np.degrees(np.arccos(np.clip(np.sum(n_new * p, axis=2), -1.0, 1.0)))

    # for angles smaller than 90°, fill matrix GI with ones, otherwise GI is zero
    # round is needed! otherwise 90 is smaller than 90
    GI = np.double(np.round(A, 2) < 89)

    # take front side of cell only
    GI[:91, :] = 0
    GI = np.flipud(GI)

    return GI

