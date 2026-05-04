# Usage

Here is a quick overview of how to use Ungrasp to load an `.sph` file.

```python
import ungrasp

# Get a sample file path (if you're running this in the source repo)
path = ungrasp.get_test_data_path('hertzian_e_dipole_x')

# Parse the .sph file
with path.open("rt") as f:
    sph_file = ungrasp.read_sph_file(f)

# Extract the frequency block
freq_block = sph_file.get(0)
print(f"Frequency: {freq_block.frequency_ghz} GHz")

# Convert to electric field
efield = ungrasp.ElectricField.from_frequency_block(freq_block)
```
