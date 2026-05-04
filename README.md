# Ungrasp — A converter for GRASP `.sph` files

This repository contains the source code of Ungrasp, a Python library that reads GRASP spherical harmonic files (`.sph`) and converts them into spin-2 Stokes a_ℓm coefficients suitable for beam convolution codes.

## Features

-   **Parsing of `.sph` files**
-   **Spin-1 to Spin-2 Transformation**: Convert the physical electric field (spin-1) computed by GRASP into Stokes parameters (I, Q, U) and their corresponding spin-0 and spin-2 ($a_{\ell m}^E$, $a_{\ell m}^B$) harmonic coefficients
-   **Polarization Projections**: Built-in support for Ludwig’s 3rd definition and standard Theta/Phi projection
-   **High-Performance Backend**: Based on Ducc0 for fast Spherical Harmonic Transforms
-   **Automated Beam Alignment**: Include an `auto_align` method that automatically rotates the beam in harmonic space to align the peak intensity and the polarization direction of the main beam
-   **Arbitrary grid evaluation**: Evaluate the complex electric field at any specific coordinate or 1D cut.

## Installation

The easiest way to add Ungrasp to your Python code is using `uv`:

```sh
uv add ungrasp
```

## Basic Usage

- Simple example
- Visualizing a cut
- Plotting a Healpix map
- Aligning a beam

## Licensing

This project is licensed under the EUPL v1.2. See [LICENSE.txt](./LICENSE.txt).

Please note that this library depends on [Ducc](https://gitlab.mpcdf.mpg.de/mtr/ducc/-/blob/ducc0/LICENSE), which is [licensed under the GPLv2](https://gitlab.mpcdf.mpg.de/mtr/ducc/-/blob/ducc0/LICENSE). When distributed together or used as a combined work, the terms of the GPL may apply to the combination as permitted by the EUPL v1.2 compatibility clause.

## Development setup

### Prerequisites

Ensure you have `uv` installed:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation Environments

We use dependency groups to keep the environment lean. Depending on your task, sync the environment using one of the following commands:

-   Standard Development (Tests, Linting, Typing):
  
    ```sh
    uv sync --group dev
    ```

-   Visualization & Research (JupyterLab, Matplotlib, Plotting):

    ```sh
    uv sync --group dev --group visualization
    ```

-   Documentation:

    ```sh
    uv sync --group docs
    ```

-   Minimal/Production (Library only):
    
    ```sh
    uv sync
    ```

### Building the documentation

The documentation is built using Sphinx. To build the HTML manual locally, simply use Nox:

```sh
uv run nox -s docs
```

The generated HTML files will be available in the `docs/_build/html` directory. You can open `docs/_build/html/index.html` in your web browser.

Alternatively, if you prefer to build the documentation without Nox, ensure you have synced the `docs` dependency group, then run:

```sh
uv run sphinx-build -b html docs/ docs/_build/html
```

### Working with Notebooks

If you are debugging or visually inspecting results using the visualization group, we recommend using the integrated Jupyter kernel.

-   Using VS Code / Cursor:

    1.   Open a .ipynb file.

    2.   Select the kernel associated with the .venv created by uv.

    3.   If the kernel isn't detected, ensure you have run `uv sync --group visualization`.

-   Using JupyterLab:

    ```sh
    uv run jupyter lab
    ```

    Note: The visualization group includes heavy dependencies like matplotlib and jupyterlab. These are excluded from the core library installation to keep the package lightweight for end-users.


### Cleaning the workspace

If you need to remove the virtual environment and start fresh, run the following commands:

```sh
rm -rf .venv
uv sync
```