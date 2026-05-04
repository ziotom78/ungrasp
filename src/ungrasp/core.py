# -*- encoding: utf-8 -*-

from dataclasses import dataclass
import numpy as np
from enum import Enum
from typing import Callable
import ducc0
import scipy

from ungrasp import FrequencyBlock


class Polarization(Enum):
    THETA_PHI = "theta_phi"
    LUDWIG3_X = "ludwig3_x"
    LUDWIG3_Y = "ludwig3_y"


def _apply_polarization(
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    phi_grid: np.ndarray,
    polarization: Polarization,
):
    """
    Applies the polarization projection matrix.

    Args:
        e_theta, e_phi (complex array): Field components.
        phi_grid (array): Azimuthal angles in radians (must match shape of e_theta).
        polarization (Polarization): The target polarization definition.

    Returns:
        (comp1, comp2): Transformed components.
    """
    if polarization == Polarization.THETA_PHI:
        return e_theta, e_phi

    sin_phi = np.sin(phi_grid)
    cos_phi = np.cos(phi_grid)

    if polarization == Polarization.LUDWIG3_X:
        # Source aligned with X-axis (Horizontal)
        # E_co = E_ϑ * cos(φ) - E_φ * sin(φ)
        # E_cx = E_ϑ * sin(φ) + E_φ * cos(φ)
        e_co = e_theta * cos_phi - e_phi * sin_phi
        e_cx = e_theta * sin_phi + e_phi * cos_phi
        return e_co, e_cx
    elif polarization == Polarization.LUDWIG3_Y:
        # Source aligned with Y-axis (Vertical)
        # E_co = E_ϑ * sin(φ) + E_φ * cos(φ)
        # E_cx = E_ϑ * cos(φ) - E_φ * sin(φ)
        e_co = e_theta * sin_phi + e_phi * cos_phi
        e_cx = e_theta * cos_phi - e_phi * sin_phi
        return e_co, e_cx
    else:
        raise NotImplementedError(f"Polarization {polarization} not supported")


MapCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]


class MapMode:
    """
    A collection of static mapping functions to transform complex electric field
    components into scalar values for visualization and analysis.

    This class serves as a namespace for standard transformations. Each method
    takes the complex :math:`E_\\theta` and :math:`E_\\phi` components (as
    NumPy arrays) and returns a single real-valued array of the same shape.
    """

    @staticmethod
    def intensity(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """
        Calculate the total power intensity of the field.

        Formula: :math:`I = |E_\\theta|^2 + |E_\\phi|^2`
        """
        return np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2

    @staticmethod
    def amplitude(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """
        Calculate the total magnitude of the electric field vector.

        Formula: :math:`A = \\sqrt{|E_\\theta|^2 + |E_\\phi|^2}`
        """
        return np.sqrt(np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2)

    @staticmethod
    def phase_theta(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Calculate the phase angle of the ϑ component in radians."""
        return np.angle(e_theta)

    @staticmethod
    def phase_phi(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Calculate the phase angle of the :math:`\\phi` component in radians."""
        return np.angle(e_phi)

    @staticmethod
    def re_theta(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Extract the real part of the ϑ component."""
        return np.real(e_theta)

    @staticmethod
    def im_theta(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Extract the imaginary part of the ϑ component."""
        return np.imag(e_theta)

    @staticmethod
    def re_phi(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Extract the real part of the φ component."""
        return np.real(e_phi)

    @staticmethod
    def im_phi(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """Extract the imaginary part of the φ component."""
        return np.imag(e_phi)

    @staticmethod
    def db(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
        """
        Calculate the intensity in decibels (dB), normalized to the peak
        value within the provided field arrays.

        Formula: :math:`10 \\log_{10}(I / I_{\\text{max}})`
        """
        int_map = MapMode.intensity(e_theta, e_phi)
        return 10 * np.log10(int_map / (np.max(int_map) + 1e-20))

    # Internal registry for the 'to_texture' method
    _REGISTRY: dict[str, MapCallable] = {
        "intensity": intensity,
        "amplitude": amplitude,
        "phase_theta": phase_theta,
        "phase_phi": phase_phi,
        "re_theta": re_theta,
        "im_theta": im_theta,
        "re_phi": re_phi,
        "im_phi": im_phi,
        "db": db,
    }

    @classmethod
    def list_modes(cls) -> list[str]:
        """Return a list of all registered mapping mode names."""
        return list(cls._REGISTRY.keys())


class ElectricField:
    """
    An electric field represented using a set of spin-1 spherical harmonics.

    This class is the core of the library, providing the bridge between the raw
    spherical wave expansion coefficients from GRASP and a manipulable,
    physical representation of the beam. It stores the field as a set of
    spin-1 spherical harmonic coefficients (:math:`a_{\\ell m}`) and provides methods
    to transform, project, and analyze the beam.

    The coefficients are derived from the GRASP :math:`Q_{smn}` coefficients and
    normalized to represent the far-field electric field vector. This class
    serves as the foundation for all subsequent operations, including conversion
    to Stokes parameters, rotation, and visualization.

    Attributes:
        frequency_ghz (float): The frequency of the monochromatic field in GHz.
        lmax (int): The maximum multipole order :math:`\\ell` of the expansion.
        mmax (int): The maximum azimuthal order :math:`m` of the expansion.
        alm_stack (np.ndarray): A NumPy array of shape `(4, nalm)` containing
            the complex spherical harmonic coefficients. These represent the
            real and imaginary parts of the electric and magnetic potentials.
    """

    def __init__(
        self,
        frequency_ghz: float,
        lmax: int,
        mmax: int,
        alm_stack: np.ndarray,
    ) -> None:
        self.frequency_ghz = frequency_ghz
        self.lmax = lmax
        self.mmax = mmax
        self.alm_stack = alm_stack

        assert self.lmax >= self.mmax

    @classmethod
    def from_frequency_block(cls, freq_block: FrequencyBlock):
        """
        Create an ElectricField instance from a FrequencyBlock.

        This class method converts the raw :math:`Q_{smn}` coefficients from a
        `ungrasp.FrequencyBlock` into the spin-1 spherical harmonic
        coefficients (:math:`a_{\\ell m}`) that represent the electric field.
        The coefficients are normalized to ensure consistency with standard
        spherical harmonic conventions.

        Args:
            freq_block (FrequencyBlock): An instance of `ungrasp.FrequencyBlock` containing the
                GRASP spherical wave expansion coefficients for a single frequency.

        Returns:
            ElectricField: A new `ElectricField` instance initialized with the
                converted spherical harmonic coefficients.
        """
        nalm = ElectricField._num_of_alms(
            freq_block.header.lmax, freq_block.header.mmax
        )
        alm_stack = np.zeros((4, nalm), dtype=np.complex128)

        ElectricField._build_alms_from_q(
            freq_block=freq_block,
            alm_stack=alm_stack,
        )

        return cls(
            frequency_ghz=freq_block.header.frequency_ghz,
            lmax=freq_block.header.lmax,
            mmax=freq_block.header.mmax,
            alm_stack=alm_stack,
        )

    @staticmethod
    def _num_of_alms(lmax: int, mmax: int) -> int:
        return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)

    @staticmethod
    def _get_idx(ell: int, m: int, lmax: int) -> int:
        """Return the index of an a_ℓm coefficient in a Healpix/ducc0 array."""
        return m * (2 * lmax + 1 - m) // 2 + ell

    def get_alms(
        self, ell: int, m: int
    ) -> tuple[np.complex128, np.complex128, np.complex128, np.complex128]:
        idx = ElectricField._get_idx(ell, m, lmax=self.lmax)
        return tuple(self.alm_stack[:, idx])

    @staticmethod
    def _build_alms_from_q(
        freq_block: FrequencyBlock,
        alm_stack: np.ndarray,
    ) -> None:
        # This normalizes the beam to 4π
        scale_factor = np.sqrt(4 * np.pi)

        lmax = freq_block.header.lmax
        mmax = freq_block.header.mmax

        # As the Electric field is a spin-1 field, we skip ℓ=0 (the monopole)
        for ell in range(1, lmax + 1):
            for m in range(mmax + 1):
                idx = ElectricField._get_idx(ell, m, lmax=lmax)

                q1_mpos = scale_factor * freq_block.get_q(s=1, m=m, n=ell)
                q2_mpos = scale_factor * freq_block.get_q(s=2, m=m, n=ell)
                q1_mneg = scale_factor * freq_block.get_q(s=1, m=-m, n=ell)
                q2_mneg = scale_factor * freq_block.get_q(s=2, m=-m, n=ell)

                j_ell = (-1j) ** ell  # (−j)ⁿ
                j_ell_1 = j_ell * (-1j)  # (−j)ⁿ⁺¹

                phase_sym = (-1) ** (ell + m)

                alm_stack[0, idx] = (
                    0.5 * j_ell * (q2_mpos + phase_sym * np.conj(q2_mneg))
                )
                alm_stack[1, idx] = (
                    -0.5 * j_ell_1 * (q1_mpos + (-1) * phase_sym * np.conj(q1_mneg))
                )

                alm_stack[2, idx] = (
                    0.5 * j_ell_1 * (q2_mpos - phase_sym * np.conj(q2_mneg))
                )
                alm_stack[3, idx] = (
                    0.5 * j_ell * (q1_mpos - (-1) * phase_sym * np.conj(q1_mneg))
                )

    def project_to_gl(
        self, n_theta: int | None = None, n_phi: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project the spherical harmonic expansion over a Gauss-Legendre grid

        Return a tuple `(E_theta, E_phi)` containing the components of the far-field
        electric vector decomposed along the ϑ/φ axes. The two arrays `E_theta` and `E_phi`
        have shape `(N, M)`, where `N` is the number of values along the ϑ direction
        and `M` the number of values along the φ direction.
        """

        # Define the Gauss-Legendre grid
        if not n_theta:
            n_theta = self.lmax + 10

        if not n_phi:
            n_phi = 2 * n_theta

        # Real part of the phasor
        map_vec_re = ducc0.sht.synthesis_2d(
            alm=self.alm_stack[0:2],
            spin=1,
            ntheta=n_theta,
            nphi=n_phi,
            geometry="GL",
            lmax=self.lmax,
            mmax=self.mmax,
        )

        # Imaginary part of the phasor
        map_vec_im = ducc0.sht.synthesis_2d(
            alm=self.alm_stack[2:4],
            spin=1,
            ntheta=n_theta,
            nphi=n_phi,
            geometry="GL",
            lmax=self.lmax,
            mmax=self.mmax,
        )

        efield_theta = map_vec_re[0] + 1j * map_vec_im[0]
        efield_phi = map_vec_re[1] + 1j * map_vec_im[1]

        return (efield_theta, efield_phi)

    def evaluate_grid(
        self,
        theta_start_rad: float,
        theta_end_rad: float,
        ntheta: int,
        phi_start_rad: float,
        phi_end_rad: float,
        nphi: int,
        polarization: Polarization,
        epsilon: float = 1e-8,
        use_grasp_phase: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the electric field on an arbitrary 2D grid of spherical coordinates.

        This method projects the spherical harmonic coefficients onto a user-defined
        grid in real space, allowing for flexible sampling of the beam. The output
        components depend on the chosen polarization basis.

        Args:
            theta_start_rad (float): Starting colatitude (polar angle) in radians (0 to pi).
            theta_end_rad (float): Ending colatitude (polar angle) in radians (0 to pi).
            ntheta (int): Number of samples along the colatitude (theta) direction.
            phi_start_rad (float): Starting longitude (azimuthal angle) in radians (0 to 2*pi).
            phi_end_rad (float): Ending longitude (azimuthal angle) in radians (0 to 2*pi).
            nphi (int): Number of samples along the longitude (phi) direction.
            polarization (Polarization): The polarization basis to use for the output components
                (see `ungrasp.Polarization`).
            epsilon (float, optional): Desired accuracy for the spherical harmonic transform, by default 1e-8.
            use_grasp_phase (bool, optional): If ``True``, the complex conjugate of the field components is returned
                to match TICRA GRASP convention for ``.cut`` and ``.grd`` files.
                By default ``False``.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two complex NumPy arrays, (Comp1, Comp2).
                Each array has shape ``(ntheta, nphi)`` and represents the
                field components in the specified polarization basis.
        """
        theta = np.linspace(theta_start_rad, theta_end_rad, num=ntheta)
        phi = np.linspace(phi_start_rad, phi_end_rad, num=nphi)

        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        # In this linearized representation of the directions, the φ angle varies faster than ϑ
        loc = np.stack((theta_grid.ravel(), phi_grid.ravel()), axis=-1)

        # Real part of the phasor
        map_vec_re = ducc0.sht.synthesis_general(
            alm=self.alm_stack[0:2],
            spin=1,
            lmax=self.lmax,
            mmax=self.mmax,
            loc=loc,
            epsilon=epsilon,
        )

        # Imaginary part of the phasor
        map_vec_im = ducc0.sht.synthesis_general(
            alm=self.alm_stack[2:4],
            spin=1,
            lmax=self.lmax,
            mmax=self.mmax,
            loc=loc,
            epsilon=epsilon,
        )

        e_theta = (map_vec_re[0] + 1j * map_vec_im[0]).reshape((ntheta, nphi))
        e_phi = (map_vec_re[1] + 1j * map_vec_im[1]).reshape((ntheta, nphi))

        result = _apply_polarization(
            e_theta=e_theta, e_phi=e_phi, phi_grid=phi_grid, polarization=polarization
        )
        if use_grasp_phase:
            return np.conj(result[0]), np.conj(result[1])
        else:
            return result

    def evaluate_cut(
        self,
        phi_angle_rad: float,
        theta_start_rad: float,
        theta_end_rad: float,
        ntheta: int,
        polarization: Polarization,
        epsilon=1e-8,
        use_grasp_phase: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract a 1D cut of the electric field at a constant azimuthal angle.

        This method evaluates the electric field along a specific meridian (constant phi)
        over a range of colatitudes. It's useful for analyzing beam patterns in a single plane.

        Args:
            phi_angle_rad (float): The constant azimuthal angle (longitude) in radians for the cut.
            theta_start_rad (float): Starting colatitude (polar angle) in radians (0 to pi).
            theta_end_rad (float): Ending colatitude (polar angle) in radians (0 to pi).
            ntheta (int): Number of samples along the colatitude (theta) direction for the cut.
            polarization (Polarization): The polarization basis to use for the output components
                (see `ungrasp.Polarization`).
            epsilon (float, optional): Desired accuracy for the spherical harmonic transform, by default 1e-8.
            use_grasp_phase (bool, optional): If ``True``, the complex conjugate of the field components is returned
                to match TICRA GRASP convention for ``.cut`` and ``.grd`` files.
                By default ``False``.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two complex NumPy arrays, (Comp1, Comp2).
                Each array has shape ``(ntheta,)`` and represents the
                field components along the 1D cut in the specified polarization basis.
        """

        e1, e2 = self.evaluate_grid(
            theta_start_rad=theta_start_rad,
            theta_end_rad=theta_end_rad,
            ntheta=ntheta,
            phi_start_rad=phi_angle_rad,
            phi_end_rad=phi_angle_rad,
            nphi=1,
            polarization=polarization,
            epsilon=epsilon,
            use_grasp_phase=use_grasp_phase,
        )

        return e1.flatten(), e2.flatten()

    def to_texture(
        self,
        shape: tuple[int, int] = (512, 1024),
        mode: str | MapCallable = MapMode.intensity,
        polarization: Polarization = Polarization.THETA_PHI,
    ) -> np.ndarray:
        """
        Generate an equirectangular projection of the electric field.

        Convert the spherical harmonic representation in a scalar 2D grid,
        sampling the field over the whole sphere and applying a mapping
        function, e.g., intensity, phase, dB.

        The grid covers the whole interval:
        - ϑ (elevation): from 0 to π (North → South)
        - φ (azimut): from 0 to 2π

        Args:
            shape (tuple[int, int], optional): Texture resolution ``(n_theta, n_phi)``.
            mode (str | MapMode | Callable): Define how to transform the
                complex components of the field (:math:`E_\\theta, E_\\phi`) into
                scalar values. It can either be a member of :class:`MapMode`,
                a string (``db``, ``phase_theta``, etc.), or a custom function
                that accepts the two NumPy arrays ``e_theta`` and ``e_phi`` and
                returns a NumPy array.
            polarization (Polarization, optional): The polarization basis to
                use to calculate the field.

        Returns:
            np.ndarray: 2D array of ``float64`` with size `shape`.

        Example:
            .. code-block:: python

                # Get a representation of the intensity of the field in dB
                texture_db = efield.to_texture(shape=(400, 800), mode="db")

                # Use a custom function to map the data to a scalar
                my_map = lambda et, ep: np.abs(et) / (np.abs(ep) + 1e-10)
                ratio_texture = efield.to_texture(mode=my_map)
        """
        n_theta, n_phi = shape
        e_theta, e_phi = self.evaluate_grid(
            0, np.pi, n_theta, 0, 2 * np.pi, n_phi, polarization
        )

        map_func: MapCallable

        if isinstance(mode, str):
            if mode in MapMode._REGISTRY:
                map_func = MapMode._REGISTRY[mode]
            else:
                available = ", ".join(f"'{m}'" for m in MapMode.list_modes())
                raise ValueError(
                    f"Invalid mode '{mode}'. Available modes are: {available}. "
                    "You can also pass a custom callable."
                )
        else:
            map_func = mode

        return map_func(e_theta, e_phi)

    def show_3d(
        self,
        shape: tuple[int, int] = (300, 600),
        mode: str | Callable = MapMode.intensity,
        polarization: Polarization = Polarization.THETA_PHI,
    ):
        """
        Render an interactive 3D visualization of the electric field on a sphere.

        This method projects the electric field components onto a spherical mesh
        and opens an interactive Plotly session. It is optimized for Jupyter
        environments to allow real-time rotation, zooming, and inspection of
        beam features like sidelobes and phase patterns.

        Args:
            shape (tuple[int, int], optional): The resolution in pixels of the spherical mesh as ``(n_theta, n_phi)``.
                Higher values increase detail but may impact rendering performance.
            mode (str | MapMode | Callable, optional): The mapping function used to convert complex field components
                (:math:`E_\\theta, E_\\phi`) into scalar values. Accepts:

                - A `MapMode` enum member (e.g., `MapMode.DB`).
                - A string key (e.g., "db", "phase_theta").
                - A custom callable: `f(e_theta, e_phi) -> scalar_array`.

            polarization (Polarization, optional): The polarization basis used to evaluate the field (e.g., THETA_PHI,
                LUDWIG3_X).

        Returns:
            plotly.graph_objects.Figure: An interactive Plotly Figure object. In Jupyter environments,
                returning this object will render the widget in the cell output.

        Notes:
            - This method requires `plotly` to be installed.
            - If ripples or high-frequency features are not visible, try increasing
              the `shape` resolution to improve the sampling density of the mesh.
            - To ensure proper rendering in JupyterLab, a kernel restart might be
              required if WebGL context issues occur.

        Example:
            .. code-block:: python

                # Visualize the beam intensity in dB
                efield.show_3d(mode="db")

                # Inspect the phase of the Ludwig-3 X-polarized component
                from ungrasp import MapMode, Polarization
                efield.show_3d(mode=MapMode.PHASE_THETA, polarization=Polarization.LUDWIG3_X)
        """
        try:
            import plotly.graph_objects as go  # ty: ignore[unresolved-import]
        except ImportError:
            print(
                "Plotly not found. Install it using 'uv add --group visualization plotly'"
            )
            return

        data = self.to_texture(shape=shape, mode=mode, polarization=polarization)

        # Generate the sphere
        theta = np.linspace(0, np.pi, shape[0])
        phi = np.linspace(0, 2 * np.pi, shape[1])
        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        X = np.sin(THETA) * np.cos(PHI)
        Y = np.sin(THETA) * np.sin(PHI)
        Z = np.cos(THETA)

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    surfacecolor=data,
                    colorscale="Viridis",
                    colorbar=dict(title=str(mode)),
                )
            ]
        )

        # Make the box a cube, so that the sphere doesn’t look like an ellipsoid
        fig.update_layout(
            title=f"Ungrasp 3D: {str(mode)}",
            scene=dict(
                aspectmode="data",  # This keeps the sphere spherical
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        return fig.show()

    def rotate(
        self, psi_rad: float, theta_rad: float, phi_rad: float
    ) -> "ElectricField":
        """
        Return a rotated copy of the beam. The rotation is expressed
        using Euler angles (Z-Y-Z convention).

        Args:
            psi_rad (float): Rotation around Z axis (first).
            theta_rad (float): Rotation around new Y axis.
            phi_rad (float): Rotation around new Z axis (last).

        Returns:
            ElectricField: A new, rotated field object.
        """

        alm_rotated = ducc0.sht.rotate_alm(
            alm=self.alm_stack,
            lmax=self.lmax,
            mmax_in=self.mmax,
            mmax_out=self.mmax,
            psi=psi_rad,
            theta=theta_rad,
            phi=phi_rad,
        )

        return ElectricField(
            frequency_ghz=self.frequency_ghz,
            lmax=self.lmax,
            mmax=self.mmax,
            alm_stack=alm_rotated,
        )

    def find_peak(
        self,
        region_theta_rad: tuple[float, float, int] = (0, np.radians(10), 30),
        region_phi_rad: tuple[float, float, int] = (0, np.radians(360), 60),
    ) -> tuple[float, float, float]:
        """
        Finds the direction of maximum intensity and its polarization twist.

        Performs a grid search followed by an optimization to locate the
        peak of the beam. It then evaluates the Ludwig-3 cross-polarization
        ratio at the peak to determine the polarization orientation.

        Args:
            region_theta_rad (tuple[float, float, int], optional): The search region for the elevation angle as (start, end, samples).
            region_phi_rad (tuple[float, float, int], optional): The search region for the azimuthal angle as (start, end, samples).

        Returns:
            tuple[float, float, float]: The coordinates of the peak and its polarization twist in radians:
                (theta_peak, phi_peak, psi_pol).
        """

        # Sample the region where the maximum is
        theta_start_rad, theta_end_rad, ntheta = region_theta_rad
        phi_start_rad, phi_end_rad, nphi = region_phi_rad
        e_co, e_cx = self.evaluate_grid(
            theta_start_rad=theta_start_rad,
            theta_end_rad=theta_end_rad,
            ntheta=ntheta,
            phi_start_rad=phi_start_rad,
            phi_end_rad=phi_end_rad,
            nphi=nphi,
            polarization=Polarization.LUDWIG3_X,
        )

        intensity = np.abs(e_co) ** 2 + np.abs(e_cx) ** 2

        # Find the rough position of the maximum
        region_theta = np.linspace(theta_start_rad, theta_end_rad, ntheta)
        region_phi = np.linspace(phi_start_rad, phi_end_rad, nphi)

        idx_max = np.argmax(intensity)
        theta_max_idx, phi_max_idx = np.unravel_index(idx_max, intensity.shape)
        theta_0 = region_theta[theta_max_idx]
        phi_0 = region_phi[phi_max_idx]

        # Use SciPy to find the accurate position of the maximum
        def objective(coords: tuple[np.float64, np.float64]) -> np.float64:
            cur_theta, cur_phi = coords
            new_theta, new_phi = self.evaluate_grid(
                theta_start_rad=cur_theta,
                theta_end_rad=cur_theta,
                ntheta=1,
                phi_start_rad=cur_phi,
                phi_end_rad=cur_phi,
                nphi=1,
                polarization=Polarization.LUDWIG3_X,
            )
            return -(np.abs(new_theta[0]) ** 2 + np.abs(new_phi[0]) ** 2)

        res = scipy.optimize.minimize(
            objective,
            x0=(theta_0, phi_0),
            bounds=[(theta_start_rad, theta_end_rad), (phi_start_rad, phi_end_rad)],
            method="Powell",
        )
        theta_peak, phi_peak = res.x

        # Create a copy of this field and rotate it so that the maximum is
        # aligned with +Z
        reoriented_beam = self.rotate(
            psi_rad=-phi_peak, theta_rad=-theta_peak, phi_rad=0.0
        )

        # Align the polarization axis with the x axis
        # (The polarization axis is the direction of the copolar component of the
        # electric field)

        e_co, e_cx = reoriented_beam.evaluate_grid(
            theta_start_rad=0.0,
            theta_end_rad=0.0,
            ntheta=1,
            phi_start_rad=0.0,
            phi_end_rad=0.0,
            nphi=1,
            polarization=Polarization.LUDWIG3_X,
        )
        psi_pol = float(np.angle(e_co[0, 0] + 1j * e_cx[0, 0]))

        return float(theta_peak), float(phi_peak), psi_pol

    def get_alignment_angles(
        self,
        region_theta_rad: tuple[float, float, int] = (0, np.radians(10), 30),
        region_phi_rad: tuple[float, float, int] = (0, np.radians(360), 60),
    ) -> dict[str, float]:
        """
        Compute the Euler angles required to center the beam and align its polarization.

        This function finds the beam's peak and calculates the inverse Z-Y-Z
        Euler rotation needed to bring the peak to the +Z axis and align the
        copolar direction with the +X axis.

        Returns:
            dict[str, float]: A dictionary containing the keys `psi_rad`, `theta_rad`, and `phi_rad`.
                This can be unpacked directly into the :meth:`.rotate` method.

        Example:
            .. code-block:: python

                angles = efield.get_alignment_angles()
                aligned_efield = efield.rotate(**angles)
        """
        theta, phi, psi = self.find_peak(region_theta_rad, region_phi_rad)

        # The inverse of a beam at (theta, phi) with twist (psi)
        return {"psi_rad": -phi, "theta_rad": -theta, "phi_rad": -psi}

    def align(
        self,
        region_theta_rad: tuple[float, float, int] = (0, np.radians(10), 30),
        region_phi_rad: tuple[float, float, int] = (0, np.radians(360), 60),
    ) -> "ElectricField":
        """Convenience method that finds the peak and returns a re-aligned copy."""
        angles = self.get_alignment_angles(region_theta_rad, region_phi_rad)
        return self.rotate(**angles)


@dataclass
class Beam:
    """
    A beam pattern decomposed into Stokes parameters (I, Q, U) via spherical harmonics
    (Spin-0 and Spin-2).

    Unlike :py:class:`ungrasp.ElectricField` which uses physical components
    (:math:`E_\\theta, E_\\phi`), this class provides a representation suitable for
    CMB data analysis and beam convolution libraries. The field is described
    using the standard CMB convention:

    - :math:`a_{\\ell m}^I`: Spin-0 harmonic coefficients representing the total intensity (Stokes I).

    - :math:`a_{\\ell m}^E`: Spin-2 harmonic coefficients (E-mode) representing gradient-like polarization.

    - :math:`a_{\\ell m}^B`: Spin-2 harmonic coefficients (B-mode) representing curl-like polarization.
    """

    alm_i: np.ndarray
    """1D array of Stokes I harmonic coefficients (:math:`a_{\\ell m}^I`)."""

    alm_e: np.ndarray
    """1D array of Stokes Q/U E-mode harmonic coefficients (:math:`a_{\\ell m}^E`)."""

    alm_b: np.ndarray
    """1D array of Stokes Q/U B-mode harmonic coefficients (:math:`a_{\\ell m}^B`)."""

    lmax: int
    """Maximum multipole order (:math:`\\ell`) for the spherical harmonic expansion."""

    mmax: int
    """Maximum azimuthal order (:math:`m`) for the spherical harmonic expansion."""

    frequency_ghz: float | None = None
    """The frequency of the beam in GHz, if known."""

    @classmethod
    def from_electric_field(
        cls,
        electric_field: ElectricField,
        lmax: int | None = None,
        mmax: int | None = None,
    ) -> "Beam":
        """
        Convert an `ElectricField` object into a `Beam` object.

        This method projects the :math:`E_\\theta` and :math:`E_\\phi` components over a spatial
        Gauss-Legendre grid, computes the local Stokes parameters (:math:`I, Q, U`),
        and then performs a Spin-0 and Spin-2 spherical harmonic transform to
        extract the :math:`I, E, B` coefficients.

        Args:
            electric_field (ElectricField): The input electric field object.
            lmax (int | None, optional): The maximum multipole order :math:`\\ell` to compute.
                If `None`, it defaults to the `lmax` of the input field.
            mmax (int | None, optional): The maximum azimuthal order :math:`m` to compute.
                If `None`, it defaults to `lmax`.

        Returns:
            Beam: A new instance populated with the computed harmonic coefficients.
        """
        E_theta, E_phi = electric_field.project_to_gl()
        assert E_theta.shape == E_phi.shape

        stokes_I = np.array(np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2, dtype=np.float64)
        stokes_Q = np.array(np.abs(E_theta) ** 2 - np.abs(E_phi) ** 2, dtype=np.float64)
        stokes_U = np.array(2 * np.real(E_theta * np.conj(E_phi)), dtype=np.float64)

        if lmax is None:
            lmax = electric_field.lmax
        if mmax is None:
            mmax = lmax

        n_theta, n_phi = E_theta.shape

        # Get back the b_ℓm
        alm_I = ducc0.sht.analysis_2d(
            map=np.ascontiguousarray(stokes_I.reshape(1, n_theta, n_phi)),
            spin=0,
            geometry="GL",
            lmax=lmax,
        )

        alm_pol = ducc0.sht.analysis_2d(
            map=np.ascontiguousarray([stokes_Q, stokes_U]),
            spin=2,
            geometry="GL",
            lmax=lmax,
        )

        return cls(
            alm_i=alm_I[0],
            alm_e=alm_pol[0],
            alm_b=alm_pol[1],
            lmax=lmax,
            mmax=mmax,
        )

    def get_idx(self, ell: int, m: int) -> int:
        """
        Return the index of an :math:`a_{\\ell m}` coefficient given a specific (:math:`\\ell, m`) pair.

        Note that only coefficients with :math:`m \\geq 0` are stored in the object,
        so the function will raise an ``AssertionError`` if :math:`m < 0`.

        Args:
            ell (int): Multipole order (:math:`\\ell`).
            m (int): Azimuthal order (:math:`m \\geq 0`).

        Returns:
            int: The zero-based index in the underlying coefficient arrays (`alm_i`, etc.).
        """
        assert ell >= 0, "ℓ={ell} cannot be negative"
        assert m >= 0, "m={m} cannot be negative"
        assert m <= ell, f"{m=} > {ell=} cannot be greater than ℓ={ell}"
        return m * (2 * self.lmax + 1 - m) // 2 + ell

    def get_alms(self, ell: int, m: int) -> tuple[complex, complex, complex]:
        """
        Retrieve the harmonic coefficients (:math:`a_{\\ell m}^I, a_{\\ell m}^E, a_{\\ell m}^B`)
        for a given pair (:math:`\\ell, m`).

        This method supports both positive and negative values of :math:`m`. When :math:`m < 0`,
        the coefficients are conjugated and the phase symmetry factor :math:`(-1)^m` is applied.

        Args:
            ell (int): Multipole order :math:`\\ell`.
            m (int): Azimuthal order :math:`m`. Can be negative.

        Returns:
            tuple[complex, complex, complex]: A 3-element tuple containing the harmonic
                coefficients for Stokes I (Spin-0), E-mode (Spin-2), and B-mode (Spin-2).

        Raises:
            ValueError: If :math:`\\ell` is out of bounds (:math:`\\ell < 0` or :math:`\\ell > lmax`).
        """
        if not (0 <= ell <= self.lmax):
            raise ValueError(f"out-of-bounds ℓ={ell}")

        if abs(m) > min(ell, self.mmax):
            return 0j, 0j, 0j

        idx = self.get_idx(ell, abs(m))

        val_i = self.alm_i[idx]
        val_e = self.alm_e[idx]
        val_b = self.alm_b[idx]

        if m < 0:
            phase = (-1) ** abs(m)
            val_i = phase * np.conj(val_i)
            val_e = phase * np.conj(val_e)
            val_b = phase * np.conj(val_b)

        return val_i, val_e, val_b

    def angular_power_spectra(
        self, ell_start: int = 2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the angular power spectra (:math:`C_\\ell`) for I, E, and B modes.

        The angular power spectrum describes the variance of the coefficients at each
        multipole :math:`\\ell`, summing over all valid :math:`m`.

        Args:
            ell_start (int, optional): The first multipole to compute. Defaults to 2,
                which is typically the lowest meaningful order for polarized beams.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A 4-element tuple
                containing 1D arrays:

                - `ells`: The multipole sequence (from `ell_start` to `lmax`).
                - `C_ℓ^I`: Power spectrum of intensity.
                - `C_ℓ^E`: Power spectrum of E-mode polarization.
                - `C_ℓ^B`: Power spectrum of B-mode polarization.
        """
        ells = np.arange(ell_start, self.lmax + 1)
        cl_i = np.zeros_like(ells, dtype=float)
        cl_e = np.zeros_like(ells, dtype=float)
        cl_b = np.zeros_like(ells, dtype=float)

        for i, ell in enumerate(ells):
            # 1. Termine m = 0 (nessuna simmetria, contato una volta sola)
            idx_0 = self.get_idx(ell, 0)
            sum_i = np.abs(self.alm_i[idx_0]) ** 2
            sum_e = np.abs(self.alm_e[idx_0]) ** 2
            sum_b = np.abs(self.alm_b[idx_0]) ** 2

            # 2. Termini m > 0 (contati due volte per riflettere anche m < 0)
            for m in range(1, min(ell, self.mmax) + 1):
                idx = self.get_idx(ell, m)
                sum_i += 2 * (np.abs(self.alm_i[idx]) ** 2)
                sum_e += 2 * (np.abs(self.alm_e[idx]) ** 2)
                sum_b += 2 * (np.abs(self.alm_b[idx]) ** 2)

            # 3. Normalizzazione
            cl_i[i] = sum_i / (2 * ell + 1)
            cl_e[i] = sum_e / (2 * ell + 1)
            cl_b[i] = sum_b / (2 * ell + 1)

        return ells, cl_i, cl_e, cl_b
