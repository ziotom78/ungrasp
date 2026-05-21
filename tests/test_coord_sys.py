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

# -*- encoding: utf-8 -*-

import pytest
import numpy as np
from scipy.spatial.transform import Rotation

from ungrasp import EulerAngles, get_euler_from_ticra_axes, get_euler_from_ticra_angles


def test_euler_inverse_identity():
    """The inverse of no rotation is no rotation."""
    e = EulerAngles(0.0, 0.0, 0.0)
    inv = e.inverse()

    assert inv.alpha_rad == 0.0
    assert inv.beta_rad == 0.0
    assert inv.gamma_rad == 0.0


def test_euler_inverse_asymmetric():
    """
    Verify that the inverse mathematically swaps the order of the angles
    and flips their signs: (−γ, −β, −α).
    """
    e = EulerAngles(0.1, 0.2, 0.3)
    inv = e.inverse()

    assert inv.alpha_rad == pytest.approx(-0.3)
    assert inv.beta_rad == pytest.approx(-0.2)
    assert inv.gamma_rad == pytest.approx(-0.1)


def test_ticra_axes_identity():
    """
    If the TICRA local axes perfectly align with the global axes,
    the extracted Euler angles must be exactly zero.
    """
    x_axis = [1.0, 0.0, 0.0]
    y_axis = [0.0, 1.0, 0.0]

    euler = get_euler_from_ticra_axes(x_axis, y_axis)

    assert euler.alpha_rad == pytest.approx(0.0)
    assert euler.beta_rad == pytest.approx(0.0)
    assert euler.gamma_rad == pytest.approx(0.0)


def test_ticra_axes_pure_tilt():
    """
    Test a simple rotation: A pure 30° tilt down the Y axis.
    Because Z rotates toward X, the new X axis points down into the Z plane.
    """
    angle = np.radians(30.0)
    x_axis = [np.cos(angle), 0.0, -np.sin(angle)]
    y_axis = [0.0, 1.0, 0.0]

    euler = get_euler_from_ticra_axes(x_axis, y_axis)

    # This maps strictly to a β rotation.
    assert euler.alpha_rad == pytest.approx(0.0)
    assert euler.beta_rad == pytest.approx(angle)
    assert euler.gamma_rad == pytest.approx(0.0)


def test_ticra_axes_round_trip():
    """
    Inject known Z-Y-Z active Euler angles, generate the corresponding TICRA
    axes from the rotation matrix, and ensure the function recovers the angles.
    (We keep beta > 0 to prevent Scipy from wrapping to another Euler branch).
    """
    alpha_in = np.radians(15.0)
    beta_in = np.radians(45.0)
    gamma_in = np.radians(60.0)

    # Create the rotation matrix
    r = Rotation.from_euler("ZYZ", [alpha_in, beta_in, gamma_in])
    rot_matrix = r.as_matrix()

    # Extract the local TICRA axes in the base frame
    x_axis = rot_matrix[:, 0]
    y_axis = rot_matrix[:, 1]

    # Recover the angles
    euler_out = get_euler_from_ticra_axes(x_axis, y_axis)

    assert euler_out.alpha_rad == pytest.approx(alpha_in)
    assert euler_out.beta_rad == pytest.approx(beta_in)
    assert euler_out.gamma_rad == pytest.approx(gamma_in)


def test_ticra_axes_physics_reconstruction():
    """
    Even if Scipy returns a different (but mathematically equivalent) branch
    of Euler angles, converting those angles back into a matrix
    *must* perfectly reconstruct the original TICRA Cartesian axes.
    """
    x_axis = [0.921542671602641, -0.380724536501091, 0.0762097875702184]
    y_axis = [0.382189457083398, 0.924069980070799, -0.00508830288514998]

    # Extract the angles
    euler = get_euler_from_ticra_axes(x_axis, y_axis)

    # Build a rotation matrix strictly from the extracted angles
    r_reconstructed = Rotation.from_euler(
        "ZYZ", [euler.alpha_rad, euler.beta_rad, euler.gamma_rad]
    )
    matrix_reconstructed = r_reconstructed.as_matrix()

    # The reconstructed X and Y columns MUST match the inputs
    np.testing.assert_allclose(matrix_reconstructed[:, 0], x_axis, atol=1e-8)
    np.testing.assert_allclose(matrix_reconstructed[:, 1], y_axis, atol=1e-8)


def test_ticra_angles_identity():
    """If all TICRA angles are zero, the Euler angles must be exactly zero."""
    euler = get_euler_from_ticra_angles(theta_rad=0.0, phi_rad=0.0, psi_rad=0.0)

    assert euler.alpha_rad == pytest.approx(0.0)
    assert euler.beta_rad == pytest.approx(0.0)
    assert euler.gamma_rad == pytest.approx(0.0)


def test_ticra_angles_pure_theta():
    """
    A pure theta rotation (phi=0, psi=0) in TICRA corresponds strictly
    to a pure beta (Y-axis) rotation in Z-Y-Z Euler angles.
    """
    euler = get_euler_from_ticra_angles(theta_rad=np.pi / 6, phi_rad=0.0, psi_rad=0.0)

    assert euler.alpha_rad == pytest.approx(0.0)
    assert euler.beta_rad == pytest.approx(np.pi / 6)
    assert euler.gamma_rad == pytest.approx(0.0)


def test_ticra_angles_pure_phi():
    """
    A pure phi rotation (theta=0, psi=0) tests the cross-axis dependency.
    According to the manual, gamma = -phi + psi. So a positive phi
    must result in a negative gamma!
    """
    euler = get_euler_from_ticra_angles(theta_rad=0.0, phi_rad=np.pi / 4, psi_rad=0.0)

    assert euler.alpha_rad == pytest.approx(np.pi / 4)
    assert euler.beta_rad == pytest.approx(0.0)
    assert euler.gamma_rad == pytest.approx(-np.pi / 4)


def test_ticra_angles_pure_psi():
    """
    A pure psi rotation (theta=0, phi=0) maps directly to the final
    gamma (Z-axis) Euler rotation.
    """
    euler = get_euler_from_ticra_angles(theta_rad=0.0, phi_rad=0.0, psi_rad=np.pi / 3)

    assert euler.alpha_rad == pytest.approx(0.0)
    assert euler.beta_rad == pytest.approx(0.0)
    assert euler.gamma_rad == pytest.approx(np.pi / 3)


def test_ticra_angles_combined_asymmetric():
    """
    Test a fully asymmetric configuration (similar to the interferometric pair).
    Verifies the exact algebraic combination: γ = -φ + ψ.
    """
    euler = get_euler_from_ticra_angles(theta_rad=0.15, phi_rad=0.45, psi_rad=0.10)

    assert euler.alpha_rad == pytest.approx(0.45)
    assert euler.beta_rad == pytest.approx(0.15)
    # γ = −0.45 + 0.10 = -0.35
    assert euler.gamma_rad == pytest.approx(-0.35)
