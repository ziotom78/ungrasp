from typing import TextIO

import ungrasp
import gzip

import numpy as np
import numpy.testing as npt

from utils import get_reference_file_path


def load_grd_file(
    f: TextIO, expected_header: tuple[int, int, int, int]
) -> tuple[int, int, np.ndarray]:
    """Load a GRASP GRD file

    Return the a 2D array with shape ``(ntheta, nphi)`` containing the value of the
    electric field E using Hansen’s convention (+ωt).
    """
    for i in range(8):
        _ = f.readline()

    actual_header = tuple((int(x) for x in f.readline().split()))
    assert actual_header == expected_header

    for i in range(2):
        _ = f.readline()

    actual_ntheta_nphi = tuple((int(x) for x in f.readline().split()))
    nphi, ntheta, _ = actual_ntheta_nphi

    data_lines = f.readlines()
    result = np.empty((len(data_lines), 2), dtype=np.complex128)
    for i, line in enumerate(data_lines):
        e_theta_re, e_theta_im, e_phi_re, e_phi_im = [float(x) for x in line.split()]
        result[i, 0] = e_theta_re + 1j * e_theta_im
        result[i, 1] = e_phi_re + 1j * e_phi_im

    # TICRA grd files use the −ωt convention, so we must take the conjugate
    return ntheta, nphi, np.conjugate(result)


def test_asymmetric_field() -> None:
    with gzip.open(get_reference_file_path("asymmetric_swe.sph.gz"), "rt") as f:
        grasp_file = ungrasp.read_sph_file(f)
    assert grasp_file.num_of_blocks == 1

    electric_field = ungrasp.ElectricField.from_frequency_block(grasp_file.get(index=0))

    with gzip.open(get_reference_file_path("asymmetric_grid.grd.gz"), "rt") as f:
        ntheta, nphi, grid = load_grd_file(f, expected_header=(1, 1, 2, 7))

    e_theta, e_phi = electric_field.evaluate_grid(
        theta_start=0.0,
        theta_end=np.deg2rad(10.0),
        ntheta=ntheta,
        phi_start=0.0,
        phi_end=np.deg2rad(359.5),
        nphi=nphi,
        polarization=ungrasp.Polarization.THETA_PHI,
        epsilon=1e-9,
    )

    differences = np.concatenate(
        [
            np.abs(e_theta.flatten() - grid[:, 0]),
            np.abs(e_phi.flatten() - grid[:, 1]),
        ]
    )

    # We do not aim to a better value than 10⁻⁶ because
    # the `.sph` file produced by GRASP has still ~10⁻⁹ power in the
    # highest multipoles. As the beam has an overall power of ~44 W,
    # this means that the relative error in power is ~10⁻¹¹, and thus
    # the error in amplitude is of the order of 10⁻⁶.
    assert np.median(differences) < 1.0e-6, "The error on the E field is too large"

    npt.assert_allclose(e_theta.flatten(), grid[:, 0], atol=2e-5)
    npt.assert_allclose(e_phi.flatten(), grid[:, 1], atol=2e-5)
