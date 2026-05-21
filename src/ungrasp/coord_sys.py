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

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class EulerAngles:
    """
    A collection of three floating-point values representing three Euler angles (in radians):
        - alpha_rad (float): Rotation around Z axis (first).
        - beta_rad (float): Rotation around new Y axis.
        - gamma_rad (float): Rotation around new Z axis (last).
    """

    alpha_rad: float
    beta_rad: float
    gamma_rad: float

    def inverse(self) -> "EulerAngles":
        """
        Return the set of Euler angles that represent the inverse rotation
        with respect to `self`.
        """

        return EulerAngles(
            alpha_rad=-self.gamma_rad,
            beta_rad=-self.beta_rad,
            gamma_rad=-self.alpha_rad,
        )


def get_euler_from_ticra_axes(
    x_axis: np.typing.ArrayLike, y_axis: np.typing.ArrayLike
) -> EulerAngles:
    """
    Converts TICRA Cartesian axis definitions into active Z-Y-Z Euler angles.

    Args:
        x_vec: An array of 3 floats representing a normalized vector in the form ``[x, y, z]``
        y_vec: An array of 3 floats representing a normalized vector in the form ``[x, y, z]``

    Returns:
        An object of type :class:`.EulerAngles`.
    """

    x_vec = np.array(x_axis)
    assert x_vec.size == 3, f"The X axis ({x_vec}) must have 3 members"

    y_vec = np.array(y_axis)
    assert y_vec.size == 3, f"The Y axis ({y_vec}) must have 3 members"

    # The Z-axis is the cross product of X and Y
    z_vec = np.cross(x_vec, y_vec)

    # Construct the rotation matrix (columns are the local basis vectors)
    rot_matrix = np.column_stack((x_vec, y_vec, z_vec))

    # Scipy 'ZYZ' (capitalized) represents intrinsic active rotations,
    # which exactly matches the Wigner-D matrix convention in ducc0.
    rot = Rotation.from_matrix(rot_matrix)
    alpha, beta, gamma = rot.as_euler("ZYZ", degrees=False)

    return EulerAngles(alpha_rad=alpha, beta_rad=beta, gamma_rad=gamma)


def get_euler_from_ticra_angles(
    theta_rad: float,
    phi_rad: float,
    psi_rad: float,
) -> EulerAngles:
    """
    Convert TICRA (ϑ, φ, ψ) GRASP angles into active Z-Y-Z Euler angles.

    According to the TICRA Tools manual, the mapping between GRASP angles and
    intrinsic Z-Y-Z Euler angles is: (α, β, γ) = (φ, ϑ, -φ + ψ).

    Args:
        theta_rad (float): The TICRA ϑ angle, in radians
        phi_rad (float): The TICRA φ angle, in radians
        psi_rad (float): The TICRA ψ angle, in radians

    Returns:
        An object of type :class:`.EulerAngles`.
    """
    alpha = phi_rad
    beta = theta_rad
    gamma = -phi_rad + psi_rad

    return EulerAngles(alpha_rad=alpha, beta_rad=beta, gamma_rad=gamma)
