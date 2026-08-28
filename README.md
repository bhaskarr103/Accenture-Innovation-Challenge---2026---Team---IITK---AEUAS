# Accenture-Innovation-Challenge---2026---Team---IITK---AEUAS

# Digital Twin – Controlled Experiments and Prototype Development

This repository contains the controlled experiments and prototype development carried out as part of the Digital Twin research project.

The work is organized into three experimental stages:

1. **Experiment 1 – AI4I**
2. **Experiment 2 – SECOM**
3. **Experiment 3 – carDA / Worker Position Prototype**

A fourth section documents the **planned simulation and Digital Twin integration** using Unreal Engine and AirSim/COSYS.

---

## 1. Papers and Datasets Used as Reference

The experiments were developed by studying relevant research papers and using their datasets/proposed ideas as references.

### Reference Approach

The main ideas investigated across the experiments include:

- Sparse Autoencoder (SAE) based representation learning
- Original + latent feature fusion
- Multiple classifier ensembles
- Bagging and boosting based approaches
- Competence-aware / confidence-aware classifier fusion
- Class-Specific Weighted Ensemble (CSWE)
- Digital Twin integration
- Real-time perception and worker/vehicle state estimation

> **Note:** The referenced papers are used as methodological inspiration. The implementations in this repository are prototype implementations developed for the controlled experiments.

### Datasets

| Experiment | Dataset | Purpose |
|---|---|---|
| Experiment 1 | AI4I 2020 Predictive Maintenance Dataset | Machine-failure prediction prototype |
| Experiment 2 | SECOM Semiconductor Manufacturing Dataset | Manufacturing failure prediction and robustness experiments |
| Experiment 3 | carDA / factory-floor video data | Worker detection, worker selection and ground-position estimation |

---

# 2. Controlled Experiments

## Experiment 1 – AI4I

### Objective

The first controlled experiment investigates a predictive-maintenance pipeline using the **AI4I 2020 dataset**.

The objective was to reproduce and test the main ideas of the selected reference methodology, particularly:

- Data preprocessing
- Sparse Autoencoder representation learning
- Latent feature extraction
- Original + latent feature fusion
- Multiple classifiers
- Ensemble prediction
- Competence/confidence-aware fusion
- Final machine-failure prediction

### Architecture

![Experiment 1 – AI4I Prototype Architecture](images/1.png)

### Reference Architecture / Paper Concept

![Reference Paper Architecture](images/0.png)

The reference methodology provided the main conceptual basis for the architecture. In particular, the experiment investigated the use of:

- Sparse Autoencoder based latent representations
- Feature fusion
- Multiple heterogeneous classifiers
- Bagging / boosting style ensemble concepts
- Competence-aware weighted fusion
- Final binary failure prediction

### Experimental Pipeline

```text
AI4I 2020 Dataset
        ↓
Data Cleaning / Preprocessing
        ↓
Sparse Autoencoder
        ↓
Latent Features
        ↓
Original + Latent Feature Fusion
        ↓
Multiple Classifiers
        ↓
Competence / Confidence Evaluation
        ↓
Weighted Ensemble Fusion
        ↓
Machine Failure Prediction
```

### Implementation

The experiment code is available under:

```text
Controlled Experiments/
└── Experiment 1 - ai4i/
```

The experiment contains the preprocessing, representation-learning, classifier and ensemble components used to evaluate the proposed prototype.

### Evaluation Metrics

The experiment was evaluated using classification metrics including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### Results

| Model / Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | — | — | — | — | — |
| Proposed / Hybrid | — | — | — | — | — |
| Ensemble | — | — | — | — | — |

---

# Experiment 2 – SECOM

### Objective

The second controlled experiment applies the representation-learning and ensemble methodology to the **SECOM semiconductor manufacturing dataset**.

The SECOM dataset contains a large number of manufacturing sensor/process features with missing values and redundant variables. The experiment therefore focused on preprocessing, feature reduction, latent representation learning and ensemble-based failure prediction.

### Dataset

The SECOM experiment used the following data-processing stages:

```text
Raw SECOM Data
      ↓
Duplicate Column Removal
      ↓
Remove Features with >50% Missing Values
      ↓
Median Imputation
      ↓
Standardization
      ↓
Sparse Autoencoder
      ↓
Latent Features
      ↓
Original + Latent Feature Fusion
      ↓
Classifier Ensemble
      ↓
Competence / Weighted Fusion
      ↓
Failure Prediction
```

### Architecture

![Experiment 2 – SECOM Prototype Architecture](images/2.png)

### Experimental Components

The experiment investigated:

- SECOM data preprocessing
- Duplicate-feature removal
- Missing-value handling
- Feature standardization
- Sparse Autoencoder representation learning
- Latent feature extraction
- Original + latent feature fusion
- SVM
- KNN
- Gaussian Naive Bayes
- Bagging
- Ensemble prediction
- Competence / weighted fusion

### Controlled Experiments and Ablations

The SECOM study was further divided into controlled experiments to understand the contribution of different components.

#### Experiment 2C – Main Experiment

The main SECOM experiment evaluates the complete prototype pipeline.

```text
SECOM
 ↓
Preprocessing
 ↓
SAE
 ↓
Hybrid Features
 ↓
Classifier Ensemble
 ↓
Weighted / Competence-Aware Fusion
 ↓
Failure Prediction
```

#### Experiment 2D – Ablation

An ablation experiment was performed to investigate the effect of removing selected components from the complete system.

```text
Complete Model
      ↓
Remove / Modify Selected Component
      ↓
Retrain / Evaluate
      ↓
Compare Performance
```

#### Experiment 2E – Latent Feature Ablation

The latent representation contribution was separately investigated by comparing configurations with and without the learned latent features.

```text
Original Features
       vs.
Original + SAE Latent Features
       ↓
Classifier / Ensemble Evaluation
```

#### Threshold Experiment

Different decision thresholds were also investigated to understand the effect of the classification threshold on the final prediction performance.

### Implementation

The experiment code is available under:

```text
Controlled Experiments/
└── Experiment 2 - secom/
```

The folder contains the main experiment, ensemble implementation, preprocessing, SAE representation learning, ablation studies and threshold experiment.

### Evaluation Metrics

The SECOM experiments were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### Results


| Experiment / Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Main SECOM | — | — | — | — | — |
| Ensemble | — | — | — | — | — |
| Ablation | — | — | — | — | — |
| Latent Ablation | — | — | — | — | — |
| Threshold Experiment | — | — | — | — | — |

---

# Experiment 3 – carDA / Worker Position Prototype

## Objective

The third controlled experiment moves from tabular predictive-maintenance data to a **real-time computer-vision prototype for worker safety and spatial awareness**.

The objective is to identify and track the worker interacting with a moving car and estimate the worker's ground position:

```text
(X, Y, Z)
```

The prototype is designed to remain useful even when the worker becomes partially occluded by the vehicle, including situations where the worker is visible through or behind the car door/window.

### Prototype Architecture

![Experiment 3 – Worker Position Prototype](images/3.png)

### Input

The prototype uses a factory-floor video captured from the **WS10 camera view**.

Person detection and pose estimation are performed using:

```text
YOLOv11n-pose
```

### Camera Calibration

The worker's image position is converted to a ground/world position using camera calibration parameters:

- Intrinsic camera matrix `K`
- Rotation matrix `R`
- Translation vector `T`
- Camera world position

The ground plane is assumed to be:

```text
Z = 0
```

The detected worker pixel position is projected onto this ground plane to estimate the real-world worker position.

### Moving Car Tracking

The car is represented using four manually selected anchor points:

```text
C1 ───────── C2
│             │
│     CAR     │
│             │
C4 ───────── C3
```

The four anchor points are tracked using **Lucas–Kanade optical flow**.

A common translation is estimated from the tracked points so that the car maintains a rigid shape during motion.

### Worker Selection Logic

For every detected person:

1. Obtain the ankle midpoint when reliable.
2. Use the bounding-box center as a fallback when ankle information is unavailable.
3. Calculate the distance from the person to the moving car polygon.
4. Calculate bounding-box overlap with the car polygon.
5. Reject people farther than the worker-selection distance threshold.
6. Calculate a worker score.
7. Apply an overlap bonus when the person overlaps the car.
8. Select the person with the minimum score as the target worker.

Conceptually:

```text
Worker Score
=
Distance to Car
-
Overlap Bonus
```

This allows a worker who is very close to or partially inside the moving-car region to remain a valid worker candidate.

### Important Prototype Capability

A key observation from the experiment is that a person can become heavily overlapped by the moving car while still being the relevant worker.

For example:

```text
Worker outside car
       ↓
Worker approaches car
       ↓
Worker overlaps car region
       ↓
Worker becomes partially occluded
       ↓
Worker remains selected as TARGET WORKER
       ↓
Ground position is estimated
```

This is important for the intended worker–vehicle interaction scenario.

### Prototype Output

The system provides:

- Target worker bounding box
- Target worker label
- Worker ground position `(X, Y, Z)`
- Distance/gap to moving car
- Worker/car overlap percentage
- YOLO confidence
- Moving car anchor
- Car translation

### Demo Video

The following video demonstrates the working Experiment 3 prototype for worker detection, moving-car tracking, and ground-position estimation.

<p align="center">
  <video
    src="https://github.com/user-attachments/assets/1aae6dc8-ef19-4a69-bcbd-ae2d84fd1d8f"
    controls
    width="1000">
  </video>
</p>

---

# 3. Planned Simulation and Digital Twin Integration

The next stage is to connect the perception and Digital Twin components with a simulated environment.

The planned simulation environment uses:

- Unreal Engine
- AirSim / COSYS
- Python Digital Twin client
- Real-time RPC communication
- Vehicle state and pose information
- Monitoring and logging

### Simulation Architecture

![Simulation Architecture](images/4.png)

The intended communication architecture is:

```text
Unreal Engine
     ↓
AirSim / COSYS
     ↓
Python Digital Twin
     ↓
Real-Time State / Pose
     ↓
Processing / Monitoring
     ↓
Logging / Visualization
```

### Planned Live Data Flow

![Simulation Data Flow](images/5.png)

The simulation is intended to provide real-time vehicle information such as:

- Speed
- RPM
- Gear
- Vehicle position `(X, Y, Z)`
- Distance to stations
- Optional sensor data

The Python Digital Twin will receive and process this information through the simulation interface.

### Planned Monitoring Pipeline

```text
Simulated Vehicle
       ↓
AirSim / COSYS RPC
       ↓
Python Digital Twin
       ↓
Real-Time Vehicle State
       ↓
CSV Logging
       ↓
Dashboard / Visualization
       ↓
Alerts and Insights
```

### Future Direction

The planned simulation stage will provide the foundation for integrating the controlled experiments with a larger Digital Twin framework.

Potential extensions include:

- Real-time perception integration
- Camera / LiDAR / IMU sensor integration
- Worker–vehicle interaction monitoring
- Predictive maintenance integration
- Anomaly detection
- Digital Twin state synchronization
- Multi-vehicle simulation
- Web/cloud-based monitoring

---

Yes — I understand now. You want **one clean Markdown block that you can copy directly into GitHub**, with the image placeholders already included.

Use this exactly:

````markdown
# 4. LineSight Digital Twin

The LineSight module implements a read-only Digital Twin for monitoring and
analyzing a synthetic production line.

The objective is to transform production event data into a digital view of the
production process and identify station-level performance degradation and
production bottlenecks.

The implemented pipeline is:

```text
Production Event Data
        ↓
Process Discovery
        ↓
Station-Level Analysis
        ↓
Cycle-Time Monitoring
        ↓
Bottleneck Detection
        ↓
Digital Twin Dashboard
````

### Digital Twin Dashboard

The LineSight dashboard provides a high-level view of the production system,
including event statistics, station information, process-discovery performance,
and identified bottlenecks.

![LineSight Dashboard](images/6.png)

The dashboard currently reports:

* Total production events
* Number of production stations
* Process-discovery precision
* Process-discovery recall
* Mean cycle time
* Cycle-time variation
* Relative slowdown
* Detected production bottleneck

### Bottleneck Detection

The Digital Twin analyzes the cycle-time behavior of individual production
stations to identify abnormal slowdowns.

![Bottleneck Detection](images/7.png)

In the current experiment, **Station 10** is identified as the primary
production bottleneck.

The Digital Twin reports:

* Mean cycle time: **12.24 min**
* Cycle-time variation: **1.70 min**
* Relative slowdown: **23.1%**

The simulation includes an equipment-wear fault at Station 10. The increasing
cycle-time multiplier represents gradual equipment degradation, allowing the
Digital Twin to expose its effect as a production bottleneck.

### Station and Sensor Coverage

The Digital Twin also maintains station-level monitoring information and
sensor-coverage status.

![Station and Sensor Coverage](images/8.png)

Stations can be classified according to their available monitoring coverage:

* Instrumented
* Partial
* Manual

The sensor-coverage view provides an overview of the available monitoring data
for each station and helps identify areas where additional instrumentation may
be beneficial.

### Read-Only Digital Twin Architecture

The current implementation follows a read-only architecture.

```text
Synthetic / Exported Production Data
              ↓
       Data Processing
              ↓
       Process Discovery
              ↓
      Digital Twin Model
              ↓
     Bottleneck Analysis
              ↓
        Dashboard
              ↓
      Monitoring / Insights
```

The current Digital Twin does not contain a PLC write path. It consumes
production data, maintains a digital representation of the production line,
and provides monitoring and analytical insights.

### Current Capabilities

The implemented LineSight prototype currently demonstrates:

* Production-event ingestion
* Process discovery
* Station-level cycle-time analysis
* Bottleneck identification
* Equipment-degradation analysis
* Sensor-coverage monitoring
* Digital Twin dashboard visualization
* Read-only system monitoring

### Future Direction

The LineSight Digital Twin can subsequently be extended toward real-time
integration with physical or simulated production systems.

Potential extensions include:

* Real-time event streaming
* Live equipment-state synchronization
* Predictive maintenance
* Anomaly detection
* Remaining useful life estimation
* Additional sensor integration
* Automated maintenance alerts
* Historical trend analysis
* Cloud-based Digital Twin deployment

---

# Repository Structure

The repository is organized as follows:

```text
DigitalTwin/
│
├── Controlled Experiments/
│   │
│   ├── Experiment 1 - ai4i/
│   │
│   ├── Experiment 2 - secom/
│   │
│   └── Experiment 3 - carDA/
│
├── images/
│   ├── 0.png
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   ├── 4.png
│   └── 5.png
│
└── pose_output.mp4
```

---

# Summary

The project currently progresses through three controlled experimental stages:

```text
Experiment 1
AI4I Predictive Maintenance
        ↓
Representation Learning + Ensemble
        ↓
Experiment 2
SECOM Manufacturing Failure Prediction
        ↓
Ablation + Latent Feature + Threshold Studies
        ↓
Experiment 3
Worker–Vehicle Perception Prototype
        ↓
Worker Detection + Car Tracking
        ↓
Ground Position Estimation
        ↓
Planned Digital Twin Simulation
        ↓
Unreal Engine + AirSim / COSYS + Python
```

The repository is intended to document the experimental development from **data-driven predictive maintenance**, through **manufacturing-failure experiments**, to **real-time perception and Digital Twin integration**.
