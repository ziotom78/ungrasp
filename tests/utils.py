from dataclasses import dataclass
from typing import TextIO

import numpy as np


@dataclass
class GraspGridFile:
    theta_start_rad: float
    theta_end_rad: float
    ntheta: int
    phi_start_rad: float
    phi_end_rad: float
    nphi: int
    e_field: np.ndarray


def load_grd_file(
    f: TextIO, expected_header: tuple[int, int, int, int]
) -> GraspGridFile:
    """Load a GRASP GRD file

    Return the a 2D array with shape ``(ntheta, nphi)`` containing the value of the
    electric field E using Hansen’s convention (+ωt).
    """
    for i in range(8):
        _ = f.readline()

    actual_header = tuple((int(x) for x in f.readline().split()))
    assert actual_header == expected_header

    _ = f.readline()
    phi_start_rad, theta_start_rad, phi_end_rad, theta_end_rad = [
        np.deg2rad(float(x)) for x in f.readline().split()
    ]

    actual_ntheta_nphi = tuple((int(x) for x in f.readline().split()))
    nphi, ntheta, _ = actual_ntheta_nphi

    data_lines = f.readlines()
    result = np.empty((len(data_lines), 2), dtype=np.complex128)
    for i, line in enumerate(data_lines):
        e_theta_re, e_theta_im, e_phi_re, e_phi_im = [float(x) for x in line.split()]
        result[i, 0] = e_theta_re + 1j * e_theta_im
        result[i, 1] = e_phi_re + 1j * e_phi_im

    return GraspGridFile(
        theta_start_rad=theta_start_rad,
        theta_end_rad=theta_end_rad,
        ntheta=ntheta,
        phi_start_rad=phi_start_rad,
        phi_end_rad=phi_end_rad,
        nphi=nphi,
        e_field=np.conjugate(result),  # TICRA grd files use the −ωt convention
    )
