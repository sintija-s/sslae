# Autoencoder-based Semi-Supervised Learning from Tabular Data (SSLAE)

This repository contains the implementation of SSLAE, an autoencoder-based architecture for semi-supervised learning (SSL) from tabular data, as published in the Machine Learning Journal: [Semi-supervised learning from tabular data with autoencoders: when does it work?](https://doi.org/10.1007/s10994-025-06898-8) (open access).

SSLAE trains a classifier and an autoencoder on a shared latent representation, so unlabeled examples contribute through the reconstruction loss. 

## Installation

Built and tested with Python 3.10. Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in `pyproject.toml` and `uv.lock`.

```bash
git clone https://github.com/sintija-s/sslae.git
cd sslae
uv sync
```

PyTorch Lightning selects the hardware automatically: a GPU (CUDA or Apple MPS) if one is available, otherwise the CPU.

`SSLAEClassifier` follows the scikit-learn estimator style.

Examples can be found in [`dummy_data_example.ipynb`](dummy_data_example.ipynb) and [`openml_example.ipynb`](openml_example.ipynb)


<!-- ### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `n_intrinsic` | `"auto"` | Size of the latent space. `"auto"` uses the TwoNN estimate, or a third of the number of features if the estimate exceeds the number of features. |
| `ssl_weight` | `0.5` | Weight of the reconstruction loss. `0.0` trains on the labeled data only, ignoring the unlabeled data, and `1.0` uses the reconstruction loss only. |
| `batch_size` | `32` | Batch size. |
| `learning_rate` | `1e-3` | Learning rate for Adam. |
| `weight_decay` | `0.0` | Weight decay for Adam. |
| `num_epochs` | `100` | Maximum number of training epochs. |
| `stopping_crit` | `10` | Early stopping patience: epochs without improvement in the validation loss. |
| `dropout` | `0.0` | Dropout rate in the classifier. |
| `mode` | `"MCC"` | Only multi-class classification is implemented. `"MLC"` is accepted but has no effect, and 2-D label arrays are rejected. |
| `random_state` | `42` | Random seed. |
| `verbose` | `False` | Show a progress bar during training. Requires `rich` (see [Usage notes](#usage-notes)). | -->

## Citation

If you use SSLAE, please cite:

```bibtex
@article{Stevanoska2025,
  title = {Semi-supervised learning from tabular data with autoencoders: when does it work?},
  volume = {114},
  ISSN = {1573-0565},
  url = {http://dx.doi.org/10.1007/s10994-025-06898-8},
  DOI = {10.1007/s10994-025-06898-8},
  number = {11},
  journal = {Machine Learning},
  publisher = {Springer Science and Business Media LLC},
  author = {Stevanoska, Sintija and Levatić, Jurica and Džeroski, Sašo},
  year = {2025},
  month = Oct 
}
```
