# Streaming ML for Intrusion Detection

## Adaptive Online Machine Learning for Real-Time Network Intrusion Detection

A streaming machine-learning framework for network intrusion detection using **incremental learning, concept-drift detection, and prequential evaluation**.

The project evaluates multiple online machine-learning algorithms on the **NSL-KDD** dataset and compares their detection performance, computational latency, model size, and ability to adapt to changing network traffic.

---

## Overview

Traditional machine-learning-based Intrusion Detection Systems (IDS) are generally trained offline using historical network traffic. Once deployed, these models may gradually become less effective as network behaviour, applications, users, and attack strategies evolve.

This project explores **streaming (online) machine learning**, where network traffic is processed continuously and models are updated incrementally as new instances arrive.

The framework evaluates five streaming classifiers:

- **Hoeffding Tree (HT)**
- **Hoeffding Adaptive Tree (HAT)**
- **Adaptive Random Forest (ARF)**
- **Online Naive Bayes**
- **Streaming k-NN**

Concept drift is investigated using:

- **ADWIN** — Adaptive Windowing
- **DDM** — Drift Detection Method
- **EDDM** — Early Drift Detection Method

The models are evaluated using **prequential (test-then-train) evaluation**, where each incoming instance is first classified and then used to update the corresponding model.

---

## Key Features

- Streaming network traffic simulation
- Online / incremental machine learning
- Multiple streaming classifiers
- Prequential evaluation
- Concept-drift detection
- Rolling accuracy analysis
- Cumulative F1-score analysis
- Drift-recovery analysis
- Per-instance latency measurement
- Model-size comparison
- Adaptive intrusion detection
- Extensible alert and prevention architecture

---

## System Architecture

```text
┌──────────────────────┐
│    NSL-KDD Dataset   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Data Preprocessing │
│ Encoding + Labelling │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Stream Simulator   │
│ Instance-by-instance │
└──────────┬───────────┘
           │
           ▼
┌────────────────────────────────────────────┐
│             Streaming ML Models            │
│                                            │
│ HT │ HAT │ ARF │ Naive Bayes │ Streaming k-NN │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌──────────────────────┐
│  Prequential Testing │
│    Predict → Train   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Concept Drift     │
│  ADWIN / DDM / EDDM  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Metrics & Analysis │
│ Accuracy / F1 /      │
│ Latency / Recovery   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Alert / Prevention   │
│      Mechanism       │
└──────────────────────┘
```

---

## Streaming Models

| Model | Description |
|---|---|
| **Hoeffding Tree** | Incremental decision tree designed for high-volume data streams |
| **Hoeffding Adaptive Tree** | Adaptive Hoeffding Tree with drift-aware learning |
| **Adaptive Random Forest** | Adaptive ensemble of online decision trees |
| **Online Naive Bayes** | Lightweight incremental probabilistic classifier |
| **Streaming k-NN** | Incremental nearest-neighbour classifier |

All models process the same traffic stream under the same evaluation procedure.

---

## Dataset

The current implementation uses the **NSL-KDD intrusion-detection dataset**.

The original attack labels are converted into a binary classification problem:

```text
Normal  →  0
Attack  →  1
```

Categorical features are encoded before being passed to the streaming models.

### Potential Additional Datasets

The framework can be extended to additional datasets such as:

- CICIDS2017
- UNSW-NB15
- Edge-IIoTset
- Other network-flow datasets

---

## Prequential Evaluation

The project uses **prequential evaluation**, also known as **test-then-train evaluation**.

For each incoming network-traffic instance:

```text
Incoming Instance
       │
       ▼
   Prediction
       │
       ▼
 Record Metrics
       │
       ▼
 Detect Drift
       │
       ▼
  Update Model
       │
       ▼
 Next Instance
```

This approach better represents the behaviour of a continuously operating online IDS than a conventional static train/test split.

---

# Results

## Baseline Performance

The baseline experiment was performed using **30,000 instances** from the NSL-KDD stream.

| Model | Accuracy | Precision | Recall | F1 | Latency (ms/instance) | Model Size |
|---|---:|---:|---:|---:|---:|---:|
| Hoeffding Tree | 92.01% | 0.888 | 0.949 | 0.917 | 0.65 | 8.6 MB |
| Hoeffding Adaptive Tree | 94.12% | 0.941 | 0.932 | 0.937 | 0.84 | 8.6 MB |
| Adaptive Random Forest | **98.42%** | **0.991** | 0.975 | **0.983** | 2.19 | 34.1 MB |
| Online Naive Bayes | 57.58% | 0.998 | 0.091 | 0.167 | 0.99 | 0.9 MB |
| Streaming k-NN | 97.67% | 0.975 | **0.976** | 0.975 | 6.79 | 1.3 MB |

### Key Observations

- **Adaptive Random Forest** achieved the highest overall performance with **98.42% accuracy** and an **F1-score of 0.983**.
- **Streaming k-NN** achieved **97.67% accuracy**, but had the highest latency at **6.79 ms per instance**.
- **Hoeffding Adaptive Tree** outperformed the standard Hoeffding Tree.
- **Online Naive Bayes** achieved very high precision but extremely low recall, indicating that it missed a significant number of attacks.
- **Adaptive Random Forest** provided the strongest predictive performance but also had the largest model footprint.

---

## Baseline Visualizations

### Rolling Accuracy

Rolling accuracy with a window of 1,000 instances shows how the five streaming models perform throughout the stream.

<p align="center">
  <img src="./result/baseline_accuracy.png" alt="Rolling Accuracy of Streaming Models" width="900">
</p>

### Cumulative F1-Score

The cumulative F1-score provides a long-term view of intrusion-detection performance throughout the stream.

<p align="center">
  <img src="./result/baseline_f1.png" alt="Cumulative F1 Score of Streaming Models" width="900">
</p>

---

# Concept Drift Detection

Network traffic is not stationary. Changes in applications, users, network conditions, and attack strategies can alter the underlying data distribution.

This phenomenon is known as **concept drift**.

The project investigates:

- **ADWIN** — Adaptive Windowing
- **DDM** — Drift Detection Method
- **EDDM** — Early Drift Detection Method

## Drift Experiment

The drift experiment uses:

```text
20,000 KDDTrain+ instances
             │
             ▼
       Drift Boundary
             │
             ▼
22,544 KDDTest+ instances
```

The transition between the two datasets is treated as a distribution shift for experimental evaluation.

## Preliminary Drift Results

| Model | Accuracy Before Drift | Lowest Accuracy After Drift | Recovery | First ADWIN Alarm |
|---|---:|---:|---:|---:|
| Hoeffding Tree | 96.25% | 83.40% | 7,250 instances | 160 |
| Hoeffding Adaptive Tree | 97.55% | 82.60% | 7,250 instances | 96 |
| Adaptive Random Forest | **98.15%** | **86.80%** | **2,250 instances** | 64 |
| Online Naive Bayes | 54.00% | 37.40% | Not recovered | 864 |
| HT + ADWIN Reset | 80.90% | 61.00% | 250* | 448 |

> **Note:** The reset-based model had already degraded before the drift boundary, so its recovery value is not directly comparable with the other models.

### Drift Observations

- Tree-based models experienced a significant performance drop after the distribution shift.
- **Adaptive Random Forest recovered fastest** among the main streaming models.
- **ADWIN detected the major distribution change relatively quickly.**
- Drift detectors can generate alarms during early model learning as well as during genuine distribution changes.
- Completely resetting a model can discard useful learned information.
- Gradual adaptation strategies may therefore be preferable to full model resets.

---

## Concept Drift Visualization

The following visualization shows rolling accuracy during the distribution shift and ADWIN drift alarms for the Hoeffding Tree.

<p align="center">
  <img src="./result/drift_accuracy.png" alt="Concept Drift Detection and Rolling Accuracy" width="900">
</p>

---

# Metrics

The framework evaluates both predictive performance and computational characteristics.

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score

## Streaming Metrics

- Rolling accuracy
- Cumulative F1-score
- Drift detection time
- Drift-recovery time
- Number of drift alarms

## Computational Metrics

- Per-instance latency
- Serialized model size
- Approximate memory footprint

---

# Installation

## Requirements

Recommended environment:

- Python 3.9+
- 8 GB RAM minimum
- 16 GB RAM recommended
- Windows or Linux
- Internet connection for initial dataset download

## Clone the Repository

```bash
git clone https://github.com/<username>/streaming-ml-intrusion-detection.git
cd streaming-ml-intrusion-detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## 1. Baseline Models

Run the baseline experiment using 30,000 instances:

```bash
python 01_baseline_models.py 30000
```

For the complete available stream:

```bash
python 01_baseline_models.py 125973
```

The experiment evaluates:

- Hoeffding Tree
- Hoeffding Adaptive Tree
- Adaptive Random Forest
- Online Naive Bayes
- Streaming k-NN

## 2. Concept Drift Detection

Run the drift detection experiment:

```bash
python 02_drift_detection.py 20000
```

The experiment evaluates:

- ADWIN
- DDM
- EDDM

and analyses model behaviour before and after the distribution shift.

---

# Repository Structure

```text
streaming-ml-intrusion-detection/
│
├── 01_baseline_models.py
├── 02_drift_detection.py
├── requirements.txt
├── README.md
│
├── result/
│   ├── baseline_accuracy.png
│   ├── baseline_f1.png
│   └── drift_accuracy.png
│
└── data/
    └── NSL-KDD/
```

---

# Result Files

| File | Description |
|---|---|
| `result/baseline_accuracy.png` | Rolling accuracy of the five streaming models |
| `result/baseline_f1.png` | Cumulative F1-score of the five streaming models |
| `result/drift_accuracy.png` | Rolling accuracy and drift behaviour |

---

# Technologies

| Technology | Purpose |
|---|---|
| **Python** | Core implementation |
| **River** | Streaming machine learning |
| **NumPy** | Numerical computation |
| **Pandas** | Data processing |
| **Scikit-learn** | Supporting ML utilities |
| **Matplotlib** | Visualization |
| **Seaborn** | Data visualization |
| **Git** | Version control |

---

# Why Streaming Machine Learning?

A conventional batch-based IDS generally follows:

```text
Historical Data
      ↓
Offline Training
      ↓
Model Deployment
      ↓
Static Predictions
      ↓
Periodic Retraining
```

A streaming IDS follows:

```text
Continuous Traffic
       ↓
   Prediction
       ↓
    Learning
       ↓
 Drift Detection
       ↓
Model Adaptation
       ↓
   Prediction
       ↓
      ...
```

Streaming learning allows models to continuously incorporate new information without requiring complete offline retraining after every change in network behaviour.

---

# Prevention Architecture

The framework is designed to support a lightweight prevention and alerting layer:

```text
              Network Traffic
                     │
                     ▼
             Streaming Model
                     │
             ┌───────┴───────┐
             │               │
           Normal          Attack
             │               │
             ▼               ▼
           Allow        Alert / Flag
                             │
                             ▼
                       Response Policy
```

Potential defensive actions include:

- Security alerts
- Suspicious-flow flagging
- Malicious connection logging
- SIEM integration
- Automated response policies
- Source blocking in controlled environments

The prevention layer should be thoroughly tested in an isolated environment before being connected to production infrastructure.

---

# Current Status

## Implemented

- [x] NSL-KDD preprocessing
- [x] Streaming data simulation
- [x] Hoeffding Tree
- [x] Hoeffding Adaptive Tree
- [x] Adaptive Random Forest
- [x] Online Naive Bayes
- [x] Streaming k-NN
- [x] Prequential evaluation
- [x] Accuracy measurement
- [x] Precision / Recall / F1 evaluation
- [x] Latency measurement
- [x] Model-size measurement
- [x] Rolling accuracy visualization
- [x] Cumulative F1 visualization
- [x] ADWIN integration
- [x] DDM integration
- [x] EDDM integration
- [x] Concept-drift experiment

---

# Future Improvements

## Drift-Detector Optimization

Potential improvements include:

- Drift-detector parameter optimization
- Improved drift adaptation
- Background learners
- Complete-stream repeated experiments
- Statistical significance analysis

## Dataset Expansion

Support for:

- CICIDS2017
- UNSW-NB15
- Edge-IIoTset
- Additional network-flow datasets

## Real-Time Deployment

Potential future capabilities:

- Real network-stream integration
- Real-time monitoring dashboard
- SIEM integration
- Automated prevention module

---

# Limitations

The current implementation has several limitations:

1. **NSL-KDD is a benchmark dataset** and does not represent every characteristic of modern production networks.
2. The stream is **simulated** rather than generated from a live network.
3. The current drift experiment uses a **predefined transition between datasets**.
4. Results can vary with stream ordering and model configuration.
5. The reported results are based on experimental runs and should not be interpreted as universal benchmarks.
6. Serialized model size is used as a proxy for model footprint and does not represent exact runtime memory consumption.
7. The prevention component is intended for controlled experimental environments.

---

# Future Work

Future development will focus on improving both adaptability and real-world applicability.

## Adaptive Learning

Potential approaches include:

- Background learners
- Adaptive ensemble weighting
- Dynamic window management
- Selective forgetting
- Continual learning
- Incremental feature selection

## Advanced Drift Detection

Future experiments can investigate:

- Page-Hinkley
- Statistical drift detection
- Conformal drift detection
- Adaptive window methods
- Ensemble drift detectors

## Real-Time Deployment

A future deployment architecture could use:

```text
Network Capture
      ↓
Feature Extraction
      ↓
Kafka / Streaming Queue
      ↓
Online IDS
      ↓
Drift Detection
      ↓
Threat Classification
      ↓
Alert / Response
```

---

# Security Disclaimer

This project is intended for **defensive cybersecurity research, experimentation, and intrusion-detection development**.

Automated prevention mechanisms should be tested in controlled environments before deployment.

The project should not be used to disrupt, block, or interfere with networks without proper authorization.

---

# Author

**Chirag Sharma**

B.Tech — Artificial Intelligence & Data Science

---

# Acknowledgements

This project builds upon research and open-source developments in:

- Streaming machine learning
- Online learning
- Concept drift detection
- Adaptive ensemble learning
- Network intrusion detection
- Cybersecurity analytics

The **River** streaming machine-learning framework is used for implementing the online learning algorithms and evaluation pipeline.
