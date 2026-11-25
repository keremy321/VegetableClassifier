# Five-class vegetable classifier

Compare a small PyTorch CNN trained from scratch with one ImageNet-pretrained ResNet18 experiment. Both models use the order **Brinjal, Broccoli, Carrot, Potato, Tomato**. The local Gradio app uses the model with the lower selected validation loss. [Türkçe README](README.md)

## Windows setup

Run these PowerShell commands from the project directory. The tested interpreter is 64-bit Python 3.14.0 on Windows 11. These commands install the CPU build for local inference.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

For GPU training or inference, use a fresh environment and replace the CPU PyTorch install line with the CUDA 13.0 command below. This build was verified on the RTX 5070 Ti Laptop GPU with NVIDIA driver 617.42.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
```

## Data and experiment

Put the original `train`, `validation`, and `test` folders from the [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) in `data/vegetables/`. If `outputs/dataset_manifest.csv` is missing, run `.\.venv\Scripts\python.exe -m src.audit`. The manifest preserves the supplied split counts for the five selected classes. Its pixel hashes support exact duplicate checks. Near duplicates may still exist.

The original photos belong in `data/personal/` with the existing `labels.csv`. They form a previously inspected external diagnostic set. Model selection uses validation loss and happens before running the dataset test or original photo evaluation.

```powershell
.\.venv\Scripts\python.exe -m src.inspect_inputs
.\.venv\Scripts\python.exe -m src.train_resnet --smoke
.\.venv\Scripts\python.exe -m src.train_resnet
.\.venv\Scripts\python.exe -m src.compare --select-only
.\.venv\Scripts\python.exe -m src.evaluate --model resnet18
.\.venv\Scripts\python.exe -m src.evaluate_personal --model resnet18 --set original
.\.venv\Scripts\python.exe -m src.compare
```

ResNet18 downloads torchvision's `IMAGENET1K_V1` weights on the first run into `.work/torch/` by default. The experiment uses 224 × 224 bilinear letterboxing, RGB mean padding, and the weights' ImageNet normalization. The training history records augmentation, optimizer settings, both stages, and the checkpoint hash. The full train command replaces the ResNet18 checkpoint. Repeat validation selection before rerunning evaluation or restarting the app after training a replacement.

The baseline configuration and compact results are in `baseline/`. The paired benchmark and validation selection use the separately distributed baseline checkpoint. To train the small CNN from scratch on the same data, run `.\.venv\Scripts\python.exe -m src.train`. That run writes to `outputs/baseline_reproduction/`. To re-evaluate the archived baseline checkpoint, use `--model baseline` with `src.evaluate` or `src.evaluate_personal`. Those reruns write to `outputs/baseline_rerun/`.

## Launch the application

Place the ResNet18 checkpoint at `checkpoints/resnet18/best.pt`. The [weight manifest](MODEL_WEIGHTS.md) lists the planned release filenames, checksums, and installation paths for both models.

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:7860`, upload a JPEG, PNG, or WebP of a supported vegetable, and select **Classify photograph**. The interface names the loaded model, and the terminal prints its checkpoint path and hash prefix. The five softmax scores rank the available classes. Use the following command to compare the baseline app on a separate local port.

```powershell
.\.venv\Scripts\python.exe app.py --checkpoint baseline/checkpoints/best.pt --port 7861
```

## Optional fresh photo evaluation

After the models are finalized, put newly collected photographs in `data/personal_fresh/` and fill in [labels.csv](data/personal_fresh/labels.csv) with one `filename,true_class` row per photo. Use one of the five class names above. The original five photos stay in their existing folder.

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_personal --model resnet18 --set fresh
.\.venv\Scripts\python.exe -m src.evaluate_personal --model baseline --set fresh
```

Fresh results go to `outputs/resnet18/fresh/` and `outputs/baseline_rerun/fresh/`. Capture a range of viewpoints, object sizes, backgrounds, and lighting if the goal is to estimate performance beyond the original setup.

## Results and report

| Path | Contents |
| --- | --- |
| `baseline/` | Small CNN configuration and saved metrics |
| `checkpoints/resnet18/best.pt` | ResNet18 checkpoint with its class mapping and preprocessing |
| `checkpoints/deployment.json` | Default model selected using validation losses |
| `outputs/input_diagnostics/input_checks.json` | Inference checks |
| `outputs/model_comparison.json` | Test, per-class, timing, and original diagnostic comparisons |
| `outputs/resnet18/` | Training history, test measurements, and predictions |
| [Turkish report](report/rapor_tr.pdf) | Submission report with the baseline and follow-up results |
