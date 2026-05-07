Usage
=====

This page provides an overview of how to use Ungrasp to load, process, and convert spherical harmonic expansions of antenna beams produced by TICRA GRASP.

1. Basic Information
--------------------

The first step is to load a `.sph` file and inspect its contents. Ungrasp allows you to parse the file and extract the frequency blocks, which contain the :math:`Q_{smn}` coefficients.

.. doctest::

    >>> import ungrasp
    >>>
    >>> # Get a sample file path (included in the Ungrasp test suite)
    >>> path = ungrasp.get_test_data_path('gaussian_beam')
    >>>
    >>> # Parse the .sph file
    >>> with path.open("rt") as f:
    ...     sph_file = ungrasp.read_sph_file(f)
    >>>
    >>> # Extract the frequency block
    >>> freq_block = sph_file.get(0)
    >>>
    >>> print(f"Frequency: {freq_block.frequency_ghz:.2f} GHz")
    Frequency: 90.00 GHz
    >>> print(f"Maximum multipole (lmax): {freq_block.lmax}")
    Maximum multipole (lmax): 50
    >>> print(f"Maximum azimuthal mode (mmax): {freq_block.header.mmax}")
    Maximum azimuthal mode (mmax): 50
    >>> print(f"Cumulative power: {freq_block.cum_power:.4f} W")
    Cumulative power: 0.5000 W


2. Evaluating an Electric Field Cut
-----------------------------------

Once the raw coefficients are loaded, we can convert them into a physical :py:class:`ungrasp.ElectricField` object. This allows us to sample the complex electric field vector along any grid or 1D cut.

In this example, we extract a 1D cut of the field at a constant azimuthal angle (:math:`\phi = 0^\circ`) and print the complex co-polar and cross-polar components using Ludwig's 3rd polarization definition.

.. doctest::

    >>> import numpy as np
    >>>
    >>> # Convert the raw coefficients to a manipulable Electric Field
    >>> efield = ungrasp.ElectricField.from_frequency_block(freq_block)
    >>>
    >>> # Define the cut parameters
    >>> phi_cut_rad = np.radians(0.0)
    >>> theta_start_rad = np.radians(0.0)
    >>> theta_end_rad = np.radians(5.0)
    >>> num_samples = 5
    >>>
    >>> # Evaluate the cut
    >>> e_co, e_cx = efield.evaluate_cut(
    ...     phi_angle_rad=phi_cut_rad,
    ...     theta_start_rad=theta_start_rad,
    ...     theta_end_rad=theta_end_rad,
    ...     ntheta=num_samples,
    ...     polarization=ungrasp.Polarization.LUDWIG3_X,
    ... )
    >>>
    >>> # Print the tabulated values
    >>> print(f"{'Theta (deg)':<15} | {'E_co (Real)':<15} | {'E_co (Imag)':<15} | {'E_cx (Real)':<15} | {'E_cx (Imag)':<15}")
    Theta (deg)     | E_co (Real)     | E_co (Imag)     | E_cx (Real)     | E_cx (Imag)
    >>> print("-" * 85)
    -------------------------------------------------------------------------------------
    >>> thetas = np.linspace(0, 5, num_samples)
    >>> for t, eco, ecx in zip(thetas, e_co, e_cx):
    ...     print(f"{t:<15.2f} | {eco.real:<15.4e} | {eco.imag:<15.4e} | {ecx.real:<15.4e} | {ecx.imag:<15.4e}")
    0.00            | 5.1627e-01      | 0.0000e+00      | 0.0000e+00      | 0.0000e+00
    1.25            | 4.6069e-01      | 0.0000e+00      | 0.0000e+00      | 0.0000e+00
    2.50            | 3.2849e-01      | 0.0000e+00      | 0.0000e+00      | 0.0000e+00
    3.75            | 1.8693e-01      | 0.0000e+00      | 0.0000e+00      | 0.0000e+00
    5.00            | 8.4907e-02      | 0.0000e+00      | 0.0000e+00      | 0.0000e+00


3. Converting to Stokes Parameters
----------------------------------

For Cosmic Microwave Background (CMB) analysis, beam convolution codes typically require the beam to be expressed in terms of Stokes parameters (I, Q, U) rather than the physical electric field.

Ungrasp provides the :py:class:`ungrasp.Beam` class, which automatically performs this conversion, decomposing the beam into Spin-0 (Intensity) and Spin-2 (E-mode and B-mode polarization) spherical harmonics.

.. doctest::

    >>> # Convert the electric field into a CMB-ready Beam object
    >>> # We limit the expansion to lmax=10 for speed
    >>> beam = ungrasp.Beam.from_electric_field(efield, lmax=10)
    >>>
    >>> # Compute the angular power spectra (C_l) for Intensity (I), E-modes, and B-modes
    >>> ells, cl_i, cl_e, cl_b = beam.angular_power_spectra()
    >>>
    >>> print(f"{'ell':<5} | {'C_ell^I':<15} | {'C_ell^E':<15} | {'C_ell^B':<15}")
    ell   | C_ell^I         | C_ell^E         | C_ell^B
    >>> print("-" * 59)
    -----------------------------------------------------------
    >>> for l, i, e, b in zip(ells[:5], cl_i[:5], cl_e[:5], cl_b[:5]):
    ...     print(f"{l:<5} | {i:<15.4e} | {e:<15.4e} | {b:<15.4e}")
    2     | 4.6738e-04      | 3.8643e-04      | 0.0000e+00
    3     | 7.2346e-04      | 5.9806e-04      | 0.0000e+00
    4     | 9.6841e-04      | 8.0061e-04      | 0.0000e+00
    5     | 1.1969e-03      | 9.8953e-04      | 0.0000e+00
    6     | 1.4042e-03      | 1.1609e-03      | 0.0000e+00
