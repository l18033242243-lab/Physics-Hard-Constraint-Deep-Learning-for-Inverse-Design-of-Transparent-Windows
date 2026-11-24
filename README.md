# Physics-Hard-Constraint Deep Learning for Inverse Design of Transparent Windows with Ultraviolet Shielding and Near-Infrared Thermal Blocking

Code, pretrained neural networks, and optical data accompanying the manuscript.

## Contents

- `13layers_with_thickness/`: 13-unit inverse-design training and evaluation, pretrained inverse models, a forward surrogate accepting material and thickness inputs, and training data.
- `Variable_layer_count_without_thickness/`: inverse models for 8–20 design units, with five training runs per unit count, a shared forward surrogate, and an evaluation notebook. Model names end with the training-run number.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Model weights and the training dataset are stored with Git LFS. Install Git LFS before cloning; in an existing clone, run:

```bash
git lfs install
git lfs pull
```

## Evaluation

Run each notebook from its own directory so that its relative input paths resolve correctly.

- For the 13-unit models, open `13layers_with_thickness/test.ipynb` and set `model_path` to the desired checkpoint.
- For the variable-unit models, open `Variable_layer_count_without_thickness/evaluate_inverse_models.ipynb`, set `N_UNITS` to a value from 8 to 20, and run the cell. It evaluates all five inverse models at that count, displays AVT, UVBR and NIRBR, and plots their TMM-calculated spectra.

The variable-unit evaluation assigns unit thicknesses using the 520 nm half-wave condition and calculates normal-incidence spectra for air / multilayer / semi-infinite glass (n = 1.5). AVT is unweighted over 400–800 nm; UVBR and NIRBR use solar-energy weighting over 280–400 nm and 800–1500 nm, respectively. The bundled `solar_spectrum.txt` uses 5 nm sampling.
