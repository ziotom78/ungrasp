# -*- encoding: utf-8 -*-
#
#  █████  █████
# ░░███  ░░███
#  ░███   ░███  ████████    ███████ ████████   ██████    █████  ████████
#  ░███   ░███ ░░███░░███  ███░░███░░███░░███ ░░░░░███  ███░░  ░░███░░███
#  ░███   ░███  ░███ ░███ ░███ ░███ ░███ ░░░   ███████ ░░█████  ░███ ░███
#  ░███   ░███  ░███ ░███ ░███ ░███ ░███      ███░░███  ░░░░███ ░███ ░███
#  ░░████████   ████ █████░░███████ █████    ░░████████ ██████  ░███████
#   ░░░░░░░░   ░░░░ ░░░░░  ░░░░░███░░░░░      ░░░░░░░░ ░░░░░░   ░███░░░
#                          ███ ░███                             ░███
#                         ░░██████                              █████
#                          ░░░░░░                              ░░░░░
#
# Copyright © 2026 Maurizio Tomasi
# This code is licensed under the EUPL 1.2
# See the file LICENSE.txt

from dataclasses import dataclass
import gzip
from typing import TextIO

import numpy as np

import ungrasp


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


def get_gaussian_beam() -> ungrasp.ElectricField:
    with gzip.open(ungrasp.get_test_data_path("gaussian_beam.sph.gz"), "rt") as f:
        grasp_file = ungrasp.read_sph_file(f)
        assert grasp_file.num_of_blocks == 1
        return ungrasp.ElectricField.from_frequency_block(grasp_file.get(index=0))
