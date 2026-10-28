# GeoGen AI

**Satellite Image Generation and Data Augmentation Using GANs** is a traditional Python project. It downloads EuroSAT RGB from its official Zenodo record, trains a class-conditional GAN in PyTorch, and serves generated images through Streamlit after a real checkpoint has been trained.

> This first stage contains source code and project documentation. It does not contain trained weights or claim any experimental results. The app intentionally refuses to generate until a checkpoint exists.

## Dataset source and licence

The authors' [EuroSAT repository](https://github.com/phelber/EuroSAT) points to the [official Zenodo record](https://zenodo.org/records/7711810), which provides `EuroSAT_RGB.zip` (about 95 MB) and `EuroSAT_MS.zip` (about 2.1 GB). The authors' README states that the dataset is under the MIT License; it also asks users to consider the [Copernicus Sentinel Data Terms](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice). This project uses the RGB archive and verifies the published MD5 checksum before extraction. Cite Helber et al. (2019) in academic work.

## Project structure

```text
GeoGen-AI/
├── app.py                   # Streamlit inference UI (requires trained checkpoint)
├── models.py                # Conditional generator and projection discriminator
├── train_gan.py             # GAN training loop with epoch checkpoints
├── dataset_utils.py         # Zenodo download, checksum, extraction, preprocessing
├── train_classifier.py      # Real-image ResNet classifier
├── evaluate.py              # Held-out metrics and sample export
├── notebooks/               # Colab training notebook
├── tests/                    # Runnable CPU smoke tests
├── docs/                     # Report, architecture, slides, viva
├── data/                     # Downloaded data (ignored by Git)
└── artifacts/                # Checkpoints, generated samples, metrics (ignored)
```

## Laptop and Colab

With 16 GB RAM and Intel integrated graphics, use the laptop for setup, data preparation, and the Streamlit app. GAN training is designed for Google Colab with a GPU runtime. CPU training is possible but likely slow. The project detects CUDA automatically and does not require an NVIDIA GPU on your laptop.

## Setup on Windows PowerShell

```powershell
cd "C:\path\to\GeoGen-AI"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download EuroSAT RGB

```powershell
python dataset_utils.py
```

The archive is stored in `data/raw/`; extracted class folders remain under `data/raw/`. Expected download size is about 95 MB.

## Run the quick smoke tests

```powershell
python -m unittest discover -s tests -v
```

These checks use tiny temporary class-folder fixtures, make one short GAN update, and exercise checkpoint loading and evaluation. They do not download EuroSAT and do not produce project results.

## Train in Google Colab

1. Upload this project folder to Google Drive as `MyDrive/GeoGen-AI` (skip `.venv`).
2. Open `notebooks/colab_training.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU** (or any available GPU).
3. Run the notebook cells from top to bottom. It downloads EuroSAT RGB, trains the same `Generator` architecture used by `app.py`, checks the checkpoint by loading it, and creates one PNG sample.
4. The final notebook cell downloads both the checkpoint and sample PNG to your laptop. Put `conditional_dcgan_latest.pt` at `artifacts/checkpoints/conditional_dcgan_latest.pt` in this project.

For a small initial trial, run `python train_gan.py --epochs 1 --batch-size 32` in Colab. A one-epoch run is only a pipeline check, not a useful quality result.

## Run the application

After installing dependencies and placing a trained checkpoint at the path above:

```powershell
streamlit run app.py
```

The generated images come from the saved model. Without weights, the app displays setup instructions and stops. The app can download the first generated tile as a PNG.

## Classifier and evaluation

Train both classifier versions with the same seed and data split. The first run uses real training images only; the second adds generated images from the trained GAN:

```powershell
python train_classifier.py --epochs 10 --seed 42
python train_classifier.py --epochs 10 --seed 42 --gan-checkpoint artifacts/checkpoints/conditional_dcgan_latest.pt --synthetic-per-class 500
python evaluate.py
```

The evaluator compares both classifiers on the same deterministic held-out split, saves accuracy and per-class F1 comparison charts, writes measured metrics to `artifacts/metrics/evaluation_comparison.json`, and exports a small set of generated examples. It refuses mismatched class orders and records the synthetic sample count from the augmented classifier checkpoint. Record experiment settings and do not describe unrun results as findings.

## Source files

- `models.py`: 64×64 RGB conditional DCGAN; generator output is `tanh` normalized.
- `train_gan.py`: BCE-with-logits objective, Adam, deterministic seed, epoch checkpoints.
- `dataset_utils.py`: downloads official RGB archive, verifies Zenodo MD5, safely extracts, creates deterministic 90/10 split.
- `train_classifier.py`: trains a matched ResNet-18 real-only baseline or a model with GAN-generated training examples.
- `evaluate.py`: compares both models' held-out accuracy and per-class precision/recall/F1, saves comparison charts and generated tiles.

## Citation

Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(7), 2217–2226. https://doi.org/10.1109/JSTARS.2019.2918242

