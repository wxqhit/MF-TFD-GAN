# Data augmentation for automated GPR-based road inspection using multi-frequency generative modeling

## Scope

This is the code for the core module of the "Data augmentation for automated GPR-based road inspection using multi-frequency generative modeling" . The former provides the data synthesis utility used to construct pre-split target-background superposition datasets, while the latter contains the neural network components used by the MF-GAN framework, including the generator, discriminator, attention modules, and reusable convolutional blocks.

## Directory Overview

```text
dataset_tools/
└── GPR_mix.py

model/
├── GAN_model/
│   ├── generator.py
│   └── discriminator.py
├── attention/
│   ├── FCANet.py
│   └── self_attentions.py
└── basic_model/
    ├── basic_model.py
    ├── spade_model.py
    └── vit_entanglement.py
```

## Dataset Synthesis Utility

The `dataset_tools/GPR_mix.py` script implements the dataset construction procedure based on target-background superposition. Under the condition that both the target library and the background library have already been divided into training subsets and validation subsets:

```text
target/
├── train/
└── val/

bk/
├── train/
└── val/
```

During generation, samples in `target/train` are only combined with samples in `bk/train`, and samples in `target/val` are only combined with samples in `bk/val`. This design avoids post-generation splitting and reduces the risk of data leakage between training and validation subsets.

For each target file, the script randomly selects a fixed number of background files. The target array is normalized and smoothed by Gaussian filtering, the background array is normalized, and the two arrays are superimposed. The mixed data are then processed by SVD filtering and stored as input data. The corresponding label is generated from the SVD-filtered target signal.

The default command is:

```bash
python dataset_tools/GPR_mix.py --target-root target --background-root bk --output-folder data_mix
```

The output structure is:

```text
data_mix/
├── train/
│   ├── data/
│   └── label/
├── val/
│   ├── data/
│   └── label/
└── file_mapping.csv
```

The `file_mapping.csv` file records the correspondence among the split type, target file, background file, generated data file, and generated label file. This mapping is intended to support experimental traceability and reproducibility.

## Model Components

### GAN Model

The `model/GAN_model` directory defines the primary adversarial components.

`generator.py` implements a multi-branch encoder-decoder generator. The encoder contains three input paths corresponding to low-, middle-, and high-frequency representations. The middle- and high-frequency paths use SPADE-conditioned residual blocks, and multi-spectral attention is applied to fuse cross-scale features. The decoder reconstructs the final single-channel output through transposed convolutional upsampling and skip-style cross connections.

`discriminator.py` defines discriminator networks based on patch embedding and attention. The main discriminator receives two inputs, extracts token-level features through repeated self-attention blocks, and applies dual-output attention to model shared and modality-specific information. A lightweight single-input discriminator variant is also provided.

### Attention Modules

The `model/attention` directory contains attention mechanisms used by the generator and discriminator.

`self_attentions.py` provides sinusoidal two-dimensional positional embeddings, patch embedding, transformer-style self-attention, feed-forward layers, and dual-output attention. These modules support token-level representation learning and cross-input feature decomposition.

`FCANet.py` implements multi-spectral channel attention based on selected DCT frequency components. The module constructs fixed DCT filters and uses the resulting frequency responses to reweight feature channels. In the generator, this module is used to enhance cross-branch feature fusion.

### Basic Model Blocks

The `model/basic_model` directory provides reusable neural network building blocks.

`basic_model.py` includes residual convolutional blocks, bottleneck blocks, SPADE-conditioned residual blocks, downsampling modules, and upsampling modules. These components form the structural basis of the generator encoder and decoder.

`spade_model.py` implements spatially adaptive denormalization. It normalizes activations using a parameter-free normalization layer and then applies learned scale and bias terms conditioned on an auxiliary feature or segmentation-like map.

`vit_entanglement.py` provides an auxiliary Vision Transformer based disentanglement model. It includes a ViT feature extractor, a shared-transformer disentanglement module, and a multimodal wrapper that produces public and private feature representations for two input modalities.

## Reproducibility Notes

The dataset synthesis script uses an explicit random seed for background selection. Training and validation generation use separate random streams by default, while preserving the separation imposed by the pre-existing `train` and `val` folders.

The generated filenames retain both the target stem and the background stem, making each synthesized sample traceable to its source components. The accompanying CSV mapping further records absolute file paths for downstream auditing.

## Dependencies Implied by These Directories

The dataset utility relies on `numpy`, `scipy`, and the project-level SVD filtering utility imported as `utils.svd_filter.svd_fil`.

The model modules rely on `torch`, `einops`, and the local modules under `model`. The SPADE implementation imports `SynchronizedBatchNorm2d` from `model.basic_model.sync_batchnorm`, which should be available in the execution environment or supplied as part of the local model package.
