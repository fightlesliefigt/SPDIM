# SPDIM: Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG

This repository contains code accompanying the ICLR 2025 submission with the title *SPDIM: Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG*.

## File list

The following files are provided in this repository:

`demo.ipynb` A jupyter notebook demonstrating SPDIM can compensate the conditional and label shifts in EEG source-free unsupervised domain adaptation problem.

`simulations.ipynb` A jupyter notebook demonstrating SPDIM abilitiy in simulations.

`spdnets` A folder containing the SPDIM geometric deep learning framework.

`pretrained_model` A folder containing the pretrained model for the source domains, enabling immediate source-free unsupervised domain adaptation.

[`manuscript_revised.pdf`](https://github.com/fightlesliefigt/SPDIM-ICLR2025-/blob/main/manuscript_revised.pdf) A copy of the revised manuscript.

[`manuscript_submitted.pdf`](https://github.com/fightlesliefigt/SPDIM-ICLR2025-/blob/main/manuscript_submitted.pdf) A copy of the oringally submitted manuscript.

[`manuscript_diff_submitted_revised.pdf`](https://github.com/fightlesliefigt/SPDIM-ICLR2025-/blob/main/manuscript_diff_submitted_revised.pdf) A pdf highlighting the changes between the original submitted and the revised manuscript. (generated with `latexdiff`)

## Requirements

All dependencies are managed with the `conda` package manager.
Please follow the user guide to [install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) `conda`.

Once the setup is completed, the dependencies can be installed in a new virtual environment.

Open a jupyter notebook and run it.


## Motor Imagery Experiment

Motor imagery experiment incorporating with SPDIM and TSMNet. Currently a public EEG BCI datasets are supported as an example: [BNCI2015001](http://bnci-horizon-2020.eu/database/data-sets)
The [moabb](https://neurotechx.github.io/moabb/) and [mne](https://mne.tools) packages are used to download and preprocess these datasets. <br>
**Notice**: there is no need to manually download and preprocess the datasets. This is done automatically on the fly

More detail instructions  are described in the `demo.ipynb` notebook.

## Simulations

Simulations of 2-2 SPD binary classification problems using our proposed generative model.

More detail instructions  are described in the `simulations.ipynb` notebook.


