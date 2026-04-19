import ungrasp

import gzip
import numpy as np

import pytest


def get_gaussian_beam() -> ungrasp.ElectricField:
    with gzip.open(ungrasp.get_test_data_path("gaussian_beam.sph.gz"), "rt") as f:
        grasp_file = ungrasp.read_sph_file(f)
        assert grasp_file.num_of_blocks == 1
        return ungrasp.ElectricField.from_frequency_block(grasp_file.get(index=0))


def test_find_peak():
    """Verifies that find_peak accurately reports the injected coordinates."""

    gaussian_efield = get_gaussian_beam()

    theta_off = np.radians(5.0)
    phi_off = np.radians(30.0)

    # Inject the displacement (direct rotation)
    displaced_efield = gaussian_efield.rotate(
        psi_rad=0.0, theta_rad=theta_off, phi_rad=phi_off
    )

    # Run the locator
    th, ph, ps = displaced_efield.find_peak(
        region_theta_rad=(0, 2 * theta_off, 40), region_phi_rad=(0, 2 * np.pi, 80)
    )

    assert th == pytest.approx(theta_off, abs=1e-3)
    assert ph == pytest.approx(phi_off, abs=1e-3)


def test_get_alignment_angles():
    """Verifies that the alignment angles successfully re-center the beam."""

    gaussian_efield = get_gaussian_beam()

    theta_off = np.radians(8.0)
    phi_off = np.radians(120.0)

    # Displace the beam
    displaced_efield = gaussian_efield.rotate(
        psi_rad=0.0, theta_rad=theta_off, phi_rad=phi_off
    )

    # Get the correction angles as a dictionary
    correction_angles = displaced_efield.get_alignment_angles(
        region_theta_rad=(0, 2 * theta_off, 40), region_phi_rad=(0, 2 * np.pi, 80)
    )

    # Apply the correction
    realigned_efield = displaced_efield.rotate(**correction_angles)

    # Verify the new peak is exactly at the North Pole (theta = 0)
    new_th, new_ph, new_ps = realigned_efield.find_peak(
        region_theta_rad=(0, np.radians(5), 20), region_phi_rad=(0, 2 * np.pi, 20)
    )

    assert new_th == pytest.approx(0.0, abs=1e-3)
    # When theta is 0, the polarization should be perfectly aligned to X
    # At the North Pole (theta=0), because of Gimbal lock φ and ψ are degenerate
    # and any azimuthal rotation is identical to a polarization twist.
    # Therefore, the true global polarization angle is their sum
    # We use np.sin() to handle the π periodicity.
    assert np.sin(new_ph + new_ps) == pytest.approx(0.0, abs=1e-3)
