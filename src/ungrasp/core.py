from dataclasses import dataclass
import numpy as np
from pathlib import Path
import ducc0

from ungrasp import FrequencyBlock


class ElectricField:
    def __init__(self, freq_block: FrequencyBlock):
        self.frequency_ghz = freq_block.header.frequency_ghz

        self.lmax = freq_block.header.nmax
        self.mmax = freq_block.header.mmax

        nalm = self._num_of_alms(self.lmax, self.mmax)
        self.alm_E_re = np.zeros(nalm, dtype=np.complex128)
        self.alm_E_im = np.zeros(nalm, dtype=np.complex128)
        self.alm_B_re = np.zeros(nalm, dtype=np.complex128)
        self.alm_B_im = np.zeros(nalm, dtype=np.complex128)
        self._build_alms_from_q(freq_block)

    @staticmethod
    def _num_of_alms(lmax: int, mmax: int) -> int:
        return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)

    def _get_idx(self, ell: int, m: int) -> int:
        "Return the index of an a_ℓm coefficient in a Healpix/ducc0 array"
        return m * (2 * self.lmax + 1 - m) // 2 + ell

    def get_alms(
        self, ell: int, m: int
    ) -> tuple[np.complex128, np.complex128, np.complex128, np.complex128]:
        idx = self._get_idx(ell, m)
        return (
            self.alm_E_re[idx],
            self.alm_B_re[idx],
            self.alm_E_im[idx],
            self.alm_B_im[idx],
        )

    def _build_alms_from_q(self, freq_block: FrequencyBlock) -> None:
        # This normalizes the beam to 4π
        scale_factor = np.sqrt(4 * np.pi)

        # As the Electric field is a spin-1 field, we skip ℓ=0 (the monopole)
        for ell in range(1, self.lmax + 1):
            for m in range(self.mmax + 1):
                idx = self._get_idx(ell, m)

                q1_mpos = scale_factor * freq_block.get_q(s=1, m=m, n=ell)
                q2_mpos = scale_factor * freq_block.get_q(s=2, m=m, n=ell)
                q1_mneg = scale_factor * freq_block.get_q(s=1, m=-m, n=ell)
                q2_mneg = scale_factor * freq_block.get_q(s=2, m=-m, n=ell)

                j_ell = (-1j) ** ell  # (−j)ⁿ
                j_ell_1 = j_ell * (-1j)  # (−j)ⁿ⁺¹

                phase_sym = (-1) ** (ell + m)

                self.alm_E_re[idx] = (
                    0.5 * j_ell * (q2_mpos + phase_sym * np.conj(q2_mneg))
                )
                self.alm_B_re[idx] = (
                    -0.5 * j_ell_1 * (q1_mpos + (-1) * phase_sym * np.conj(q1_mneg))
                )

                self.alm_E_im[idx] = (
                    0.5 * j_ell_1 * (q2_mpos - phase_sym * np.conj(q2_mneg))
                )
                self.alm_B_im[idx] = (
                    0.5 * j_ell * (q1_mpos - (-1) * phase_sym * np.conj(q1_mneg))
                )

    def project_to_gl(self) -> tuple[np.ndarray, np.ndarray]:
        """Project the spherical harmonic expansion over a Gauss-Legendre grid

        Return a tuple `(E_theta, E_phi)` containing the components of the far-field
        electric vector decomposed along the ϑ/φ axes. The two arrays `E_theta` and `E_phi`
        have shape `(N, M)`, where `N` is the number of values along the ϑ direction
        and `M` the number of values along the φ direction.
        """

        # Define the Gauss-Legendre grid
        n_theta = self.lmax  # In previous tests, there was a  +10
        n_phi = 2 * n_theta

        # Real part of the phasor
        map_vec_re = ducc0.sht.synthesis_2d(
            alm=np.ascontiguousarray([self.alm_E_re, self.alm_B_re]),
            spin=1,
            ntheta=n_theta,
            nphi=n_phi,
            geometry="GL",
            lmax=self.lmax,
            mmax=self.mmax,
        )

        # Imaginary part of the phasor
        map_vec_im = ducc0.sht.synthesis_2d(
            alm=np.ascontiguousarray([self.alm_E_im, self.alm_B_im]),
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


@dataclass
class Beam:
    alm_i: np.ndarray  # Spin-0
    alm_e: np.ndarray  # Spin-2
    alm_b: np.ndarray  # Spin-2
    lmax: int
    mmax: int
    frequency_ghz: float | None = None

    @classmethod
    def from_electric_field(
        cls,
        electric_field: ElectricField,
        lmax: int | None = None,
        mmax: int | None = None,
    ) -> "Beam":
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

    def get_idx(self, ell: int, m: int):
        assert ell >= 0
        assert m >= 0
        assert m <= ell
        return m * (2 * self.lmax + 1 - m) // 2 + ell

    def rotate(self, theta: float, phi: float, psi: float) -> "Beam":
        raise NotImplementedError

    def cut(self, phi: float, theta_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_fits(self, file_name: str | Path, convention: str = "COSMO") -> None:
        raise NotImplementedError
