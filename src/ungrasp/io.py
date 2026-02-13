from dataclasses import dataclass
import re
from typing import TextIO

import numpy as np

FREQUENCY_HEADER_REGEXP = re.compile(r"Freq \[([^]]+)\]: +([0-9]+\.[0-9]+)")
FREQUENCY_CONVERSION = {
    "THz": 1e3,
    "GHz": 1,
    "MHz": 1e-3,
    "kHz": 1e-6,
}


class SphFormatError(Exception):
    """Raised when a .sph file is ill-formatted."""


@dataclass(frozen=True)
class SphFileHeader:
    frequency_ghz: float
    ntheta: int
    nphi: int
    nmax: int
    mmax: int

    @classmethod
    def read(cls, f: TextIO) -> "SphFileHeader":
        frequency_line = f.readline().strip()
        if frequency_line == "":
            raise EOFError

        matches = FREQUENCY_HEADER_REGEXP.search(frequency_line)
        if matches is None:
            raise SphFormatError(
                f'Unable to understand the frequency in "{frequency_line}"'
            )
        unit = matches.group(1)
        value = float(matches.group(2))

        frequency_ghz = float(value) * FREQUENCY_CONVERSION[unit]

        _ = f.readline()  # Skip this line

        header_line = f.readline()
        try:
            value_list = [int(x) for x in header_line.split()]
            assert len(value_list) == 4

            ntheta, nphi, nmax, mmax = [int(x) for x in value_list]

            assert ntheta > 0
            assert nphi > 0
            assert nmax >= 0
            assert mmax >= 0
        except (AssertionError, ValueError) as exc:
            raise SphFormatError(f'Wrong header line "{header_line}"') from exc

        # Skip all the other lines in the header
        for i in range(5):
            _ = f.readline()

        return cls(
            frequency_ghz=frequency_ghz,
            ntheta=ntheta,
            nphi=nphi,
            nmax=nmax,
            mmax=mmax,
        )

    @property
    def lmax(self) -> int:
        """Return the value of nmax

        This is a handy shorthand, because ℓ is the common symbol used in the
        CMB community in place of ``n``."""
        return self.nmax


class FrequencyBlock:
    """Store the values of Q_smn for one frequency in a GRASP .sph file"""

    @staticmethod
    def _q_array_shape(mmax: int, nmax: int) -> tuple[int, int, int]:
        return (2, nmax, 2 * mmax + 1)

    def __init__(
        self,
        header: SphFileHeader,
        q_array: np.ndarray | None = None,
        cum_power: float = 0.0,
    ):
        self.header = header
        self.cum_power = cum_power

        if q_array is None:
            self.q_array = np.zeros(
                self._q_array_shape(mmax=header.mmax, nmax=header.nmax),
                dtype=np.complex128,
            )
        else:
            assert q_array.shape == self._q_array_shape(
                mmax=header.mmax, nmax=header.nmax
            )
            self.q_array = np.array(q_array, dtype=np.complex128)

    @property
    def frequency_ghz(self) -> float:
        """Return the frequency (in GHz) of the EM radiation associated with this expansion"""
        return self.header.frequency_ghz

    def _index(self, s: int, m: int, n: int) -> tuple[int, int, int] | None:
        """Return the position of the Q value corresponding to the indexes smn.

        The position is a tuple `(s, m, n)`
        The index `s` can either be 1 or 2; index `m` goes from −n to +n,
        and `n` must be a non-negative number."""
        if n < 1 or abs(m) > n or abs(m) > self.header.mmax:
            return None

        return (s - 1, n - 1, m + self.header.mmax)

    def get_q(self, s: int, m: int, n: int) -> complex:
        """Return the Q value corresponding to the indexes smn."""
        pos = self._index(s, m, n)
        if not pos:
            return 0.0j

        return self.q_array[pos]

    def set_q(self, s: int, m: int, n: int, value: complex) -> None:
        """Return the Q value corresponding to the indexes smn."""
        pos = self._index(s, m, n)
        self.q_array[pos] = value

    @staticmethod
    def _read_n_complex(f: TextIO, n: int) -> list[complex]:
        result: list[complex] = []
        while True:
            values = [float(x) for x in f.readline().split()]
            assert len(values) % 2 == 0

            for i in range(len(values) // 2):
                re = values[2 * i]
                im = values[2 * i + 1]
                result.append(complex(re, im))

            if len(result) == n:
                break

        return result

    @classmethod
    def read(cls, f: TextIO, header: SphFileHeader) -> "FrequencyBlock":
        result = cls(
            header=header,
            cum_power=0.0,
        )
        for cur_abs_m in range(header.mmax + 1):
            mode_header = f.readline()
            mode_header_values = mode_header.split()
            if len(mode_header_values) != 2:
                raise SphFormatError(f'Invalid line "{mode_header}"')

            cur_abs_m_from_header = int(mode_header_values[0])
            assert cur_abs_m_from_header == cur_abs_m

            cur_power = float(mode_header_values[1])
            result.cum_power += cur_power

            n_start = max(1, cur_abs_m)

            if cur_abs_m == 0:
                coeffs = cls._read_n_complex(f, n=2 * header.nmax)
            else:
                coeffs = cls._read_n_complex(f, n=4 * (header.nmax - (cur_abs_m - 1)))

            coeff_iter = iter(coeffs)
            for n in range(n_start, header.nmax + 1):
                if cur_abs_m == 0:
                    result.set_q(1, 0, n, next(coeff_iter))
                    result.set_q(2, 0, n, next(coeff_iter))
                else:
                    result.set_q(1, -cur_abs_m, n, next(coeff_iter))
                    result.set_q(2, -cur_abs_m, n, next(coeff_iter))
                    result.set_q(1, +cur_abs_m, n, next(coeff_iter))
                    result.set_q(2, +cur_abs_m, n, next(coeff_iter))

        return result


class SphFile:
    """Contents of a TICRA GRASP spherical wave expansion file

    A SWE file contains one or more harmonic expansions of an electric field,
    each associated to a monochromatic frequency.
    """

    def __init__(self, frequency_blocks: list[FrequencyBlock]) -> None:
        self.frequency_blocks = frequency_blocks

    def get(self, index: int) -> FrequencyBlock:
        """Return the i-th frequency block in the file

        The first block has index 0.

        See also :class:`FrequencyBlock`.
        """
        return self.frequency_blocks[index]

    @property
    def num_of_blocks(self) -> int:
        """Return the number of blocks loaded from the file."""
        return len(self.frequency_blocks)


def read_sph_file(f: TextIO) -> SphFile:
    """
    Parse a GRASP `.sph` file

    Parse a `.sph` file created by GRASP. The file can contain multiple
    frequencies. Return a list of :class:`FrequencyBlock` objects.
    """

    blocks = []

    while True:
        try:
            header = SphFileHeader.read(f)
            block = FrequencyBlock.read(f, header)
            blocks.append(block)
        except EOFError:
            break

    return SphFile(frequency_blocks=blocks)
