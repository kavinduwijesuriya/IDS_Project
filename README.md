# Binary Intrusion Detection System
## Introduction

Traditional Intrusion Detection Systems (IDS) use predefined signatures to detect known attacks. However, they may struggle to identify new or unknown attacks.

This project explores a machine learning based Intrusion Detection System using the UNSW-NB15 dataset. It trains and compares multiple supervised learning algorithms to classify network traffic as **Normal** or **Attack**.

The main objective is to develop a lightweight and accurate model that identifies malicious network activity using network-flow features.

## Features

- Binary classification of network traffic.
- Training and validation split before preprocessing.
- Training-only encoding, scaling, feature selection, and SMOTE.
- Comparison of Decision Tree, Random Forest, and Extra Trees.
- Selection of the best model using Attack F1-score.
- Reduction from approximately 192 encoded features to 30 selected features.
- Final evaluation using the separate UNSW-NB15 testing dataset.
- Hardware and system resource profiling (RAM usage, CPU utilization, training time, inference latency, throughput, model size).
- Streamlit dashboard with record replay, playback controls, live metrics, and recent detections.

## Model Results

Random Forest achieved the best validation performance and was evaluated on the independent official UNSW-NB15 test set.

### Detection Performance

| Metric | Test Result |
|---|---:|
| Accuracy | 96.31% |
| Attack Precision | 96.14% |
| Attack Recall | 98.20% |
| Attack F1-score | 97.16% |
| False-positive Rate | 7.09% |
| Selected Features | 30 |

## Hardware & System Resource Metrics

| Resource Metric | Random Forest (Final Test) |
|---|---:|
| Inference Latency | ~0.0018 ms / sample |
| Inference Throughput | ~568,000+ samples/sec |
| Training Time | ~8.78 seconds |
| Training CPU Utilisation | ~759.8% |
| Inference CPU Utilisation | ~670.9% |
| Process RAM Usage (RSS) | ~810 MB |
| Serialized Model Size | ~215 MB |

CPU utilisation represents cumulative usage across multiple CPU cores. RAM usage represents total Python process memory.

## Project Structure

```text
new ids/
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_train_select_and_save_model.ipynb
│   └── 03_results_and_features.ipynb
├── dataset/
├── gui/
├── .streamlit/
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Dataset

Download the UNSW-NB15 dataset and place these files inside the local `dataset/` folder:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

## Training

Run the notebooks in the following order:

1. `01_data_check.ipynb` – Checks the dataset and its contents.
2. `02_train_select_and_save_model.ipynb` – Preprocesses data, selects features, compares models, trains the final model, and evaluates it.
3. `03_results_and_features.ipynb` – Displays model performance and selected features.

After training, the generated model files are saved locally for use by the dashboard.

## Dashboard

The Streamlit dashboard replays records from the testing dataset. It displays detected attacks, normal traffic, running metrics, recent alerts, and a confusion matrix. Replay results can also be downloaded as a CSV file.

### Project Scope
This project was initially developed as part of a group assignment and was further enhanced by the team with live packet capture and virtual-machine integration.
