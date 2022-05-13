# Condensate Speckle

### Anqi Chen, Wenyun Wang, Hanyue Wang

This package is designed to characterize the local motion of FUS protein condensate by extracting local decay constants out of intensity flucutation. This packaged uses AR(1) model to infer decay times. For primary goal, we are inferring single decay time and precision. For the 'can do' goal, we extract two decay timescales to probe the co-existence of liquid and gel phase.

Notebook documentation.ipynb will provide tutorial for usage of the package. All the experimental data used for inference are under condensate_speckle/example_data directory. Data are stored as csv file and are named by pixel_aged-hour_intensity.csv. Loading data functions are stored in data_io.py; functions for simulating data are stored in inference/simulation.py; functions for model inferences are stored in inference/model.py.

The package can be cloned using: (pip install git+https://github.com/phys201/condensate_speckles).

