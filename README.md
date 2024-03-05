![LOGO](https://github.com/DeepWave-KAUST/diffefwi/blob/main/asset/diffefwi.png)

Reproducible material for  **DW0028 - Mohammad H. Taufik, Fu Wang, Tariq Alkhalifah.**

# Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo.
* :open_file_folder: **data**: a folder containing the subsampled velocity models used to train the diffusion model.
* :open_file_folder: **notebooks**: reproducible notebook for the third synthetic test of the paper (near-surface SEAM Arid model).
* :open_file_folder: **saves**: a folder containing the trained diffusion model (using the combined dataset) and results from the EFWI.
* :open_file_folder: **scripts**: a set of Python scripts used to run diffusion training, diffusion sampling, and EFWI.
* :open_file_folder: **src**: a folder containing routines for the `diffefwi` source file.

## Notebooks
The following notebooks are provided:

- :orange_book: ``Example-2-efwi.ipynb``: notebook reproducing the results of the near-surface synthetic test in the paper.

## Scripts
The following scripts are provided:

- 📝: ``Example-0-unconditional-sampling.py``: drawing unconditional samples from a trained diffusion model.
- 📝: ``Example-1-diffusion-training.py``: diffusion model training using the `combined` dataset of the paper.
- 📝: ``Example-2-efwi.py``: simple multi-parameter checkerboard test with an acquisition setting mimicking the land field data application of the paper.

## Getting started :space_invader: :robot:
To ensure the reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

To install the environment, run the following command:
```
./install_env.sh
```
It will take some time, but if, in the end, you see the word `Done!` on your terminal, you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate diffefwi
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) Silver 4316 CPU @ 2.30GHz equipped with a single NVIDIA A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.


## Cite us 
```bibtex
@article{taufik2024learned,
  title={Learned regularizations for multi-parameter elastic full waveform inversion using diffusion models}, 
  doi={10.1029/2024JH000125},
  author={Taufik, Mohammad Hasyim and Wang, Fu and Alkhalifah, Tariq},
  journal={Journal of Geophysical Research: Machine Learning and Computation},
  year={2024},
  publisher={Wiley Online Library}
}

