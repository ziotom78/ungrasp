import ungrasp
import gzip

import numpy as np
import numpy.testing as npt

import pytest
from utils import load_grd_file


@pytest.mark.parametrize(
    "data_file,expected_header,polarization",
    [
        (
            "asymmetric_grid_thetaphi.grd.gz",
            (1, 1, 2, 7),
            ungrasp.Polarization.THETA_PHI,
        ),
        ("asymmetric_grid_ludwig.grd.gz", (1, 3, 2, 7), ungrasp.Polarization.LUDWIG3_X),
    ],
)
def test_asymmetric_field(
    data_file: str,
    expected_header: tuple[int, int, int, int],
    polarization: ungrasp.Polarization,
) -> None:
    # Here we read a “hard” pattern: an asymmetric main lobe with diffraction
    # sidelobes produced by a rectangular hor

    with gzip.open(ungrasp.get_test_data_path("asymmetric_swe.sph.gz"), "rt") as f:
        grasp_file = ungrasp.read_sph_file(f)
    assert grasp_file.num_of_blocks == 1

    electric_field = ungrasp.ElectricField.from_frequency_block(grasp_file.get(index=0))

    with gzip.open(ungrasp.get_test_data_path(data_file), "rt") as f:
        grid = load_grd_file(f, expected_header=expected_header)

    e_theta, e_phi = electric_field.evaluate_grid(
        theta_start_rad=grid.theta_start_rad,
        theta_end_rad=grid.theta_end_rad,
        ntheta=grid.ntheta,
        phi_start_rad=grid.phi_start_rad,
        phi_end_rad=grid.phi_end_rad,
        nphi=grid.nphi,
        polarization=polarization,
        epsilon=1e-9,
    )

    rel_differences = np.concatenate(
        [
            np.abs(e_theta.flatten() - grid.e_field[:, 0]),
            np.abs(e_phi.flatten() - grid.e_field[:, 1]),
        ]
    )

    # We do not aim to a better value than 10⁻⁵ because
    # the highest multipoles  in the `.sph` file produced by GRASP has
    # still ~10⁻⁹ power right before the truncation. As the beam has an
    # overall power of ~44 W, this means that the relative error in power
    # is ~10⁻¹⁰, and thus the error in amplitude is of the order of 10⁻⁵.
    assert np.median(rel_differences) < 1.0e-5, "The error on the E field is too large"

    npt.assert_allclose(e_theta.flatten(), grid.e_field[:, 0], atol=1e-4)
    npt.assert_allclose(e_phi.flatten(), grid.e_field[:, 1], atol=1e-4)


def load_dipole_sph(file_name: str):
    with open(ungrasp.get_test_data_path(file_name), "rt") as f:
        grasp = ungrasp.read_sph_file(f)
        return ungrasp.ElectricField.from_frequency_block(grasp.get(0))


def test_rotation_consistency():
    field_x = load_dipole_sph("hertzian_e_dipole_x.sph")
    field_y = load_dipole_sph("hertzian_e_dipole_y.sph")
    field_z = load_dipole_sph("hertzian_e_dipole_z.sph")

    # Rotation around Z: X→Y
    rotated_x = field_x.rotate(psi_rad=np.pi / 2, theta_rad=0.0, phi_rad=0.0)
    assert np.allclose(rotated_x.alm_stack, field_y.alm_stack, atol=1e-12)

    # Rotation around Y: Z → X
    rotated_z = field_z.rotate(psi_rad=0, theta_rad=np.pi / 2, phi_rad=0)
    assert np.allclose(rotated_z.alm_stack, field_x.alm_stack, atol=1e-12)
