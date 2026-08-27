# Predictive Maintenance — AI4I 2020 Prototype

## Overview

This repository contains our first experimental prototype for a predictive-maintenance system based on the **AI4I 2020 Predictive Maintenance Dataset**.

The main purpose of this work is not to claim that the current model is the final competition solution.

Instead, the AI4I experiments were used to:

1. Understand the predictive-maintenance problem.
2. Reproduce the main ideas of the paper architecture.
3. Build our own implementation rather than directly copying the paper.
4. Understand Sparse Autoencoders (SAEs).
5. Understand classifier competence and CSWE-style ensemble fusion.
6. Investigate failure-specific weighting.
7. Investigate confidence/threshold-based decision making.
8. Test whether the approach is stable across multiple random seeds.
9. Establish a working prototype that can later be transferred to the current contest dataset.

The final AI4I prototype provides us with a complete experimental pipeline:

```text
Raw sensor data
      ↓
Data preprocessing
      ↓
Sparse Autoencoder
      ↓
Latent representation
      ↓
Hybrid features
      ↓
Multiple classifiers
      ↓
Validation-based competence
      ↓
Ensemble decision
      ↓
Failure prediction
```

The important point is that this is a **prototype architecture and experimental foundation**. The same ideas can later be adapted to the actual competition dataset.

---

# 1. Dataset — AI4I 2020 Predictive Maintenance Dataset

## 1.1 Dataset selection

We initially selected the **AI4I 2020 Predictive Maintenance Dataset** because it provides a relatively small but realistic predictive-maintenance classification problem.

The dataset contains:

```text
10,000 samples
14 columns
```

The original columns are:

```text
0   UDI
1   Product ID
2   Type
3   Air temperature [K]
4   Process temperature [K]
5   Rotational speed [rpm]
6   Torque [Nm]
7   Tool wear [min]
8   Machine failure
9   TWF
10  HDF
11  PWF
12  OSF
13  RNF
```

The main target used in our experiments was:

```text
Machine failure
```

where:

```text
0 = Normal
1 = Failure
```

---

# 2. Understanding the AI4I Data

The dataset represents machine operating conditions using several physical/process measurements.

The five main numerical process variables are:

### 1. Air temperature [K]

This represents the surrounding/air temperature associated with the machine.

### 2. Process temperature [K]

This represents the machine/process temperature.

### 3. Rotational speed [rpm]

This represents the rotational speed of the machine.

### 4. Torque [Nm]

This represents the applied/generated torque.

### 5. Tool wear [min]

This represents accumulated tool wear.

These variables form the main sensor/process input used by our first prototype.

---

# 3. Target Imbalance

One of the most important properties of AI4I is its severe class imbalance.

The target distribution is:

```text
Normal  = 9661
Failure =  339
```

Therefore:

```text
Normal  = 96.61%
Failure =  3.39%
```

This has an important consequence.

A classifier could achieve approximately 96.6% accuracy simply by predicting:

```text
Normal
Normal
Normal
Normal
...
```

while detecting almost no actual failures.

Therefore, accuracy alone is not sufficient for this problem.

We therefore focused heavily on:

```text
Precision
Recall
F1 score
Confusion matrix
Number of failures correctly detected
```

especially:

```text
Failure Recall
Failure F1
```

because missing a real machine failure is much more important than simply obtaining a high overall accuracy.

---

# 4. Initial Feature Design

The original AI4I dataset contains more information than we initially wanted to use directly.

The first controlled experiment used the five numerical process variables:

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

Target:

```text
Machine failure
```

We deliberately excluded:

```text
UDI
Product ID
```

because these are identifiers rather than physical process measurements.

We also initially excluded:

```text
TWF
HDF
PWF
OSF
RNF
```

because these columns explicitly describe failure modes and therefore can make the classification task artificially easy or create information leakage relative to the sensor-only formulation we wanted to investigate.

Later we also experimented with including:

```text
Type
```

as a categorical input.

The numerical variables were standardized before being passed into the learning pipeline.

---

# 5. Train / Validation / Test Split

The AI4I dataset was divided into:

```text
Training   : 7000 samples
Validation : 1500 samples
Test       : 1500 samples
```

The split was stratified so that the failure distribution remained approximately constant.

The resulting failure rates were approximately:

```text
Training   ≈ 3.39%
Validation ≈ 3.40%
Test       ≈ 3.40%
```

This is important because the dataset is highly imbalanced.

The validation set was used for:

```text
Classifier evaluation
Competence estimation
Failure-specific weighting
Threshold learning
```

The test set was kept separate for final evaluation.

---

# 6. Why We Did Not Simply Copy the Paper

The goal was not to reproduce a paper line-by-line.

Instead, we took the **architecture idea** from the paper and implemented our own experimental version.

The important concepts we adopted were:

```text
Sparse Autoencoder
        +
Multiple classifiers
        +
Classifier competence
        +
Class-specific ensemble weighting
        +
Failure-oriented decision making
```

This allowed us to understand the architecture and then modify it experimentally.

The implementation therefore acts as a **paper-inspired prototype**, not as a claim of exact reproduction of every implementation detail in the original work.

---

# 7. Overall Model Architecture

Our AI4I pipeline became:

```text
                    AI4I Dataset
                         │
                         ▼
                Data preprocessing
                         │
                         ▼
                Standardized inputs
                         │
                         ▼
                Sparse Autoencoder
                         │
                         ▼
                  Latent features
                         │
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      Original features        SAE features
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                  Hybrid features
                         │
                         ▼
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
             SVM        KNN       GaussianNB
              │
              └──────────────┐
                             ▼
                          Bagging
                             │
                             ▼
                    Validation competence
                             │
                             ▼
                    Ensemble prediction
                             │
                             ▼
                     Machine failure
```

The architecture contains two main stages:

1. **Representation learning**
2. **Failure classification / ensemble decision**

---

# 8. Sparse Autoencoder

## 8.1 Purpose

The Sparse Autoencoder (SAE) was introduced to learn a compressed representation of the sensor data.

Instead of relying only on the original process measurements, the SAE attempts to learn a latent representation containing useful structure from the input.

The basic concept is:

```text
Input
  ↓
Encoder
  ↓
Latent representation
  ↓
Decoder
  ↓
Reconstructed input
```

The autoencoder is trained to reconstruct the original input.

---

# 9. SAE Architecture

The AI4I prototype eventually used a paper-style SAE configuration with:

```text
Input dimension : 8
Hidden dimension: 12
Latent dimension: 4
```

The input dimension became 8 when the five numerical features were combined with a three-dimensional representation of the categorical `Type` variable.

Conceptually:

```text
8 input features
       ↓
     12 neurons
       ↓
      4 latent
       ↓
     decoder
       ↓
8 reconstructed features
```

The latent representation therefore reduced the dimensionality:

```text
8 → 4
```

The latent features were then extracted and combined with the original standardized features.

---

# 10. Sparse Autoencoder Objective

The SAE was not trained only using reconstruction error.

The training objective included:

```text
Reconstruction loss
+
Sparsity penalty
+
L2 regularization
```

Conceptually:

```text
Total Loss =
    Reconstruction Loss
    +
    β × Sparsity Loss
    +
    L2 Regularization
```

The sparsity target was:

```text
ρ = 0.05
```

The sparsity weight used in the improved version was:

```text
β = 0.1
```

The purpose of the sparsity constraint was to encourage the latent neurons to remain mostly inactive and therefore learn more selective representations.

---

# 11. Why Sparsity Was Important

The SAE experiments initially showed an important problem.

In one earlier configuration, the latent representation became nearly constant.

For example, the latent activations were approximately:

```text
[0.05, 0.05, 0.05, 0.05]
```

and their standard deviations were extremely small.

This is known as a form of latent collapse.

That means:

```text
Different inputs
      ↓
Almost identical latent representation
```

which makes the latent representation almost useless for classification.

We therefore adjusted the sparsity configuration.

The improved SAE produced latent activations such as:

```text
[0.0685, 0.0711, 0.0584, 0.0606]
```

and the latent representation had meaningful variation.

This was an important debugging step because it showed us that simply adding an SAE is not enough; the representation has to actually contain information.

---

# 12. Hybrid Representation

After training the SAE, the latent features were extracted.

The classifier input was then constructed as:

```text
Original standardized features
+
SAE latent features
```

For the final AI4I configuration:

```text
Original = 8
Latent   = 4
Hybrid   = 12
```

Therefore:

```text
Hybrid feature vector = 12 dimensions
```

This was our attempt to combine:

```text
Direct sensor information
+
Learned nonlinear representation
```

rather than forcing the classifier to use only the compressed representation.

---

# 13. Multiple Classifiers

Instead of relying on a single classifier, we trained four different classifiers:

```text
1. SVM
2. KNN
3. Gaussian Naive Bayes
4. Bagging
```

The reason for using multiple classifiers was that different models have different strengths.

For example, during validation:

```text
SVM
→ relatively conservative

KNN
→ somewhat better failure detection

GaussianNB
→ higher failure sensitivity but weaker precision

Bagging
→ strongest overall failure performance
```

This diversity is important for an ensemble system.

---

# 14. Classifier Competence

The key idea behind CSWE is that different classifiers should not necessarily have equal influence.

Instead, we evaluate their performance on the validation data.

For example, one of our AI4I validation results was:

```text
Classifier       Normal F1     Failure F1
------------------------------------------
SVM                0.9850        0.2903
KNN                0.9846        0.3662
GaussianNB         0.9695        0.2414
Bagging            0.9911        0.7174
```

This tells us something important.

Bagging was much more competent at recognizing the minority failure class.

Therefore, it makes sense to assign Bagging a greater influence in failure prediction.

---

# 15. Failure-Specific Competence

We eventually moved from simply looking at overall classifier performance to specifically measuring:

```text
Failure F1
Failure Recall
```

The failure competence values in the improved AI4I experiment were:

```text
SVM          0.1797
KNN          0.2267
GaussianNB   0.1494
Bagging      0.4441
```

After normalization:

```text
SVM          0.1797
KNN          0.2267
GaussianNB   0.1494
Bagging      0.4441
```

These values effectively give the ensemble a mechanism for saying:

```text
"If Bagging is better at detecting failures,
trust its failure prediction more."
```

This was one of the central ideas we wanted to test.

---

# 16. Why This Is Better Than Simple Majority Voting

A simple majority-voting ensemble treats classifiers approximately equally:

```text
SVM       → 1 vote
KNN       → 1 vote
GNB       → 1 vote
Bagging   → 1 vote
```

But this ignores classifier specialization.

Our competence-based approach instead attempts to use:

```text
Classifier performance
        ↓
Failure-specific competence
        ↓
Different influence
```

For predictive maintenance, this is attractive because a classifier that is particularly good at identifying failures should have more influence over the failure decision.

---

# 17. `paper_cswe.py`

The first major implementation was:

```text
paper_cswe.py
```

This was our **paper-style implementation**.

Its main stages were:

```text
1. Load AI4I
2. Select input features
3. Split into train/validation/test
4. Standardize features
5. Train Sparse Autoencoder
6. Extract latent representation
7. Build hybrid features
8. Train SVM
9. Train KNN
10. Train GaussianNB
11. Train Bagging
12. Evaluate validation competence
13. Construct competence matrix
14. Generate test predictions
15. Perform CSWE-style fusion
16. Evaluate final failure prediction
```

The important purpose of this file was to establish a working implementation of the paper-inspired architecture.

---

# 18. Paper-Style CSWE Result

With the paper-style setup, the final AI4I result was approximately:

```text
Accuracy  : 97.07%
Precision : 76.92%
Recall    : 19.61%
F1        : 31.25%
```

Confusion matrix:

```text
[[1446    3]
 [  41   10]]
```

There were:

```text
51 actual failures
13 predicted failures
10 correctly detected
```

This result immediately revealed an important problem.

The system achieved high overall accuracy but detected only:

```text
10 / 51
```

failures.

That corresponds to:

```text
19.61% failure recall
```

So the basic paper-style implementation was **not yet good enough for our use case**.

---

# 19. Bagging Was Stronger Than CSWE

Interestingly, the individual Bagging classifier performed substantially better.

The comparison was:

```text
Method                  Accuracy   Precision   Recall    F1
----------------------------------------------------------------
Bagging                   98.00%     86.21%     49.02%   62.50%
Majority Voting           97.33%     76.19%     31.37%   44.44%
Paper CSWE                97.07%     76.92%     19.61%   31.25%
```

This was an important finding.

It showed that:

```text
More complicated ensemble
        ≠
Automatically better performance
```

Our CSWE implementation was actually making the final decision worse than Bagging.

Instead of hiding this result, we used it to identify what needed improvement.

---

# 20. Why This Was Valuable

The purpose of the experiment was not simply to maximize one metric.

We wanted to understand:

```text
Where does the architecture help?
Where does it fail?
Why does it fail?
What can we change?
```

The AI4I experiment gave us exactly that information.

We discovered:

### Finding 1

The dataset is strongly imbalanced.

### Finding 2

Accuracy is misleading.

### Finding 3

The SAE can learn a compact representation.

### Finding 4

The SAE needs carefully chosen sparsity settings.

### Finding 5

Different classifiers have very different failure behaviour.

### Finding 6

Bagging was the strongest individual classifier.

### Finding 7

Naive CSWE weighting could actually hurt failure recall.

### Finding 8

A better failure-specific decision mechanism was required.

---

# 21. Improvements After `paper_cswe.py`

We then experimented with improving the ensemble.

The important modification was to explicitly use the failure competence of each classifier.

The normalized failure weights became approximately:

```text
SVM          0.1797
KNN          0.2267
GaussianNB   0.1494
Bagging      0.4441
```

This correctly reflected the fact that Bagging was the strongest failure classifier.

The resulting normalized CSWE performance was:

```text
Accuracy  : 98.20%
Precision : 83.33%
Recall    : 58.82%
F1        : 68.97%
```

Confusion matrix:

```text
[[1443    6]
 [  21   30]]
```

This was a major improvement over the original CSWE.

Failure detection:

```text
Actual failures       = 51
Predicted failures    = 36
Correctly detected    = 30
```

Compared with the original CSWE:

```text
Original CSWE:
Predicted = 14
Detected  = 12
```

The improved competence-weighted system therefore demonstrated that the original fusion mechanism was not the end point.

---

# 22. Confidence-Aware Decision Making

We then introduced another important idea:

Instead of automatically using the default classifier probability threshold of:

```text
0.50
```

we learned a failure threshold from validation data.

The validation experiment found:

```text
Best threshold ≈ 0.28
```

The validation result at that threshold was approximately:

```text
F1       = 70.91%
Recall   = 76.47%
Precision= 66.10%
```

The resulting test performance was:

```text
Accuracy  : 97.93%
Precision : 70.83%
Recall    : 66.67%
F1        : 68.69%
```

Confusion matrix:

```text
[[1435   14]
 [  17   34]]
```

Failure detection:

```text
Actual failures      = 51
Predicted failures   = 48
Correctly detected   = 34
```

This demonstrated another important lesson:

> In a heavily imbalanced predictive-maintenance problem, the default probability threshold is not necessarily the appropriate operating point.

---

# 23. Experiment 1 — Multi-Seed Validation

After obtaining promising results, we did not want to rely on a single random train/test split.

Therefore we created:

```text
experiment1_multiseed.py
```

This was our first robustness experiment.

We evaluated the system using:

```text
Seeds:
1
2
3
4
42
```

The purpose was to determine whether the observed performance was simply caused by a lucky split.

---

# 24. Multi-Seed Results

The average results across the five seeds were:

```text
Method                             Accuracy         Precision
----------------------------------------------------------------
Bagging                       98.45 ± 0.19%     83.90 ± 4.16%

Majority Voting               97.73 ± 0.22%     82.29 ± 3.58%

Original CSWE                 97.37 ± 0.19%     92.98 ± 7.34%

Confidence-Aware CSWE         97.96 ± 0.22%     72.30 ± 7.86%
```

For recall:

```text
Bagging                       67.84 ± 7.41%

Majority Voting               42.75 ± 9.34%

Original CSWE                 25.10 ± 7.77%

Confidence-Aware CSWE         67.84 ± 13.49%
```

For F1:

```text
Bagging                       74.74 ± 4.08%

Majority Voting               55.73 ± 7.14%

Original CSWE                 38.84 ± 8.66%

Confidence-Aware CSWE         68.81 ± 5.98%
```

---

# 25. Multi-Seed Ranking

The ranking by mean F1 was:

```text
1. Bagging                  74.74%
2. Confidence-Aware CSWE   68.81%
3. Majority Voting         55.73%
4. Original CSWE           38.84%
```

This result is extremely useful because it confirms that the original paper-style CSWE was not our best-performing configuration.

Instead:

```text
Bagging
```

was the strongest and most stable individual method.

However:

```text
Confidence-Aware CSWE
```

was also competitive and provided much better failure-oriented behaviour than the original CSWE.

---

# 26. Failure Detection Across Seeds

The average number of failures detected out of the 51 test failures was:

```text
Bagging
34.60 / 51

Majority Voting
21.80 / 51

Original CSWE
12.80 / 51

Confidence-Aware CSWE
34.60 / 51
```

This reinforces why we track actual failure detection in addition to accuracy.

A system detecting:

```text
35 / 51 failures
```

is much more useful for predictive maintenance than one detecting:

```text
13 / 51 failures
```

even if both systems have high overall accuracy.

---

# 27. `experiment1_multiseed.py`

The purpose of this script was therefore different from `paper_cswe.py`.

## `paper_cswe.py`

Main purpose:

```text
Build and understand the paper-inspired architecture.
```

Pipeline:

```text
SAE
 ↓
Hybrid features
 ↓
Multiple classifiers
 ↓
Competence matrix
 ↓
CSWE
```

## `experiment1_multiseed.py`

Main purpose:

```text
Test whether the observed results are stable.
```

Pipeline:

```text
Seed 1 ──┐
Seed 2 ──┤
Seed 3 ──┤
Seed 4 ──┤
Seed 42 ─┘
          ↓
    Average performance
          ↓
    Standard deviation
```

This distinction is important.

The first script establishes the architecture.

The second script establishes experimental confidence in the results.

---

# 28. What We Actually Built

At this point, the AI4I project is no longer simply:

```text
"we trained a classifier."
```

We have built a complete predictive-maintenance prototype:

```text
                    SENSOR DATA
                         │
                         ▼
                 PREPROCESSING
                         │
                         ▼
              SPARSE REPRESENTATION
                         │
                         ▼
               HYBRID FEATURE SPACE
                         │
                         ▼
        ┌────────────────────────────────┐
        │        CLASSIFIER BANK         │
        │                                │
        │  SVM   KNN   GNB   BAGGING     │
        └────────────────────────────────┘
                         │
                         ▼
              FAILURE COMPETENCE
                         │
                         ▼
                WEIGHTED FUSION
                         │
                         ▼
             CONFIDENCE / THRESHOLD
                         │
                         ▼
               MACHINE FAILURE
```

This is the prototype architecture that we can now transfer to the current competition problem.

---

# 29. Why This Is Useful for the Current Contest

The AI4I dataset is not the final competition dataset.

That is important.

We are **not** claiming:

```text
AI4I performance = contest performance
```

Instead, AI4I gave us a controlled environment in which we could develop and debug the complete methodology.

The contest dataset can now become the real target.

---

# 30. What AI4I Taught Us Before Moving to the Contest

The biggest value of the AI4I experiment is that we already know several things that would otherwise have to be discovered during the contest.

## Lesson 1 — Accuracy is not enough

For highly imbalanced predictive maintenance:

```text
Accuracy ≠ useful failure detection
```

We therefore track:

```text
Failure Precision
Failure Recall
Failure F1
Confusion Matrix
Detected failures
```

---

## Lesson 2 — A single classifier is a useful baseline

Bagging consistently performed very well.

Therefore, any future architecture should always be compared against:

```text
Strong single-model baseline
```

Otherwise, an ensemble could appear impressive while actually being worse than a simpler model.

---

## Lesson 3 — Classifier competence matters

Different classifiers behave differently.

For example:

```text
SVM
→ conservative

KNN
→ moderate

GaussianNB
→ sensitive

Bagging
→ balanced / strong
```

This diversity can potentially be exploited rather than treating every classifier equally.

---

## Lesson 4 — Failure-specific weighting matters

If Bagging is substantially better at identifying failures, the ensemble should recognize that.

This motivated:

```text
Failure competence
        ↓
Normalized weights
        ↓
Weighted fusion
```

---

## Lesson 5 — Threshold selection matters

The default:

```text
probability >= 0.50
```

is not necessarily optimal.

Validation-based threshold selection allowed the system to move toward a better recall/F1 operating point.

This idea can be particularly useful in the competition if the evaluation metric penalizes missed failures.

---

## Lesson 6 — Representation learning must be validated

The SAE was not automatically beneficial.

We learned that:

```text
SAE
```

is a tool, not a guarantee of improvement.

For the current contest, we should test:

```text
Raw features
+
Engineered features
+
SAE features
```

and keep the representation only if it actually improves validation performance.

---

## Lesson 7 — Robustness matters

The multi-seed experiment showed why one lucky split is insufficient.

We therefore have a template for future experiments:

```text
Run multiple seeds
        ↓
Calculate mean
        ↓
Calculate standard deviation
        ↓
Compare methods
```

---

# 31. Why This Is a Prototype Rather Than the Final System

The current AI4I system is deliberately a prototype.

There are several things we have not yet solved completely:

```text
1. Dataset-specific feature engineering
2. Advanced imbalance handling
3. Time-series/degradation modelling
4. More sophisticated ensemble selection
5. Better probability fusion
6. Calibration methodology
7. Cross-validation for the final system
8. Cost-sensitive decision making
9. Deployment/real-time inference
```

Those are future stages.

The purpose of AI4I was to make sure the fundamental architecture works end-to-end before applying it to the actual competition data.

---

# 32. Proposed Transfer to the Current Contest

The current contest pipeline can follow the structure we developed:

```text
                CONTEST DATASET
                       │
                       ▼
              DATA UNDERSTANDING
                       │
                       ▼
            DATA QUALITY ANALYSIS
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
       Missing values       Class imbalance
             │                   │
             └─────────┬─────────┘
                       ▼
                  PREPROCESSING
                       │
                       ▼
               FEATURE ENGINEERING
                       │
                       ▼
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
  Original features             SAE features
        │                             │
        └──────────────┬──────────────┘
                       ▼
                  CLASSIFIERS
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         SVM          KNN         GNB
                       │
                       ▼
                    Bagging
                       │
                       ▼
             FAILURE COMPETENCE
                       │
                       ▼
                WEIGHTED FUSION
                       │
                       ▼
            VALIDATION THRESHOLD
                       │
                       ▼
                FINAL PREDICTION
```

The important difference is that on the contest dataset we should **not assume** the AI4I configuration is optimal.

The contest data must determine:

```text
feature selection
SAE architecture
latent dimension
classifier choice
competence weighting
threshold
```

---

# 33. Current Prototype Philosophy

The main principle going forward is:

> **Experiment first, then lock the architecture based on evidence.**

For every component we should ask:

```text
Does it improve validation performance?

Does it improve failure recall?

Does it improve F1?

Is the improvement stable across seeds?

Does it beat the simpler baseline?
```

If the answer is no, we remove it.

This is exactly what happened with the AI4I SAE experiments and the original CSWE formulation.

---

# 34. Current AI4I Baselines

The most useful AI4I reference results from our experiments are:

### Bagging

Multi-seed:

```text
Accuracy : 98.45 ± 0.19%
Recall   : 67.84 ± 7.41%
F1       : 74.74 ± 4.08%
```

### Confidence-Aware CSWE

Multi-seed:

```text
Accuracy : 97.96 ± 0.22%
Recall   : 67.84 ± 13.49%
F1       : 68.81 ± 5.98%
```

### Majority Voting

```text
Accuracy : 97.73 ± 0.22%
Recall   : 42.75 ± 9.34%
F1       : 55.73 ± 7.14%
```

### Original CSWE

```text
Accuracy : 97.37 ± 0.19%
Recall   : 25.10 ± 7.77%
F1       : 38.84 ± 8.66%
```

---

# 35. Main Conclusion From AI4I

The most important conclusion is not:

> "CSWE achieved X%."

Instead, the conclusion is:

> We successfully developed and experimentally validated a predictive-maintenance pipeline combining representation learning, multiple classifiers, failure-specific competence, and validation-based decision thresholds.

The experiments demonstrated that:

```text
Bagging
```

was the strongest baseline,

while:

```text
Confidence-Aware CSWE
```

provided competitive failure detection and demonstrated the potential value of competence-aware fusion.

At the same time, the original paper-style CSWE implementation was weaker than the Bagging baseline, showing that ensemble complexity must be justified experimentally rather than assumed to improve performance.

---

# 36. Why This Matters for the Contest

The contest is now no longer a completely unknown problem.

We already have a tested experimental framework.

Instead of starting from:

```text
Dataset
 ↓
???
```

we start from:

```text
Dataset
 ↓
Data analysis
 ↓
Strong baseline
 ↓
Representation experiments
 ↓
Classifier bank
 ↓
Competence analysis
 ↓
Failure-aware fusion
 ↓
Threshold optimization
 ↓
Multi-seed validation
```

That is the real value of the AI4I prototype.

It gives us a **research and engineering framework** that can be transferred and adapted to the contest dataset.

---

# 37. Current Project Status

```text
AI4I 2020
   │
   ├── Dataset understanding                    ✓
   │
   ├── Imbalance analysis                      ✓
   │
   ├── Feature preprocessing                   ✓
   │
   ├── Sparse Autoencoder                      ✓
   │
   ├── Latent representation                   ✓
   │
   ├── Hybrid features                         ✓
   │
   ├── Multiple classifiers                    ✓
   │
   ├── Competence matrix                       ✓
   │
   ├── Paper-style CSWE                       ✓
   │
   ├── Failure-specific weighting              ✓
   │
   ├── Confidence threshold                    ✓
   │
   ├── Multi-seed validation                   ✓
   │
   └── Prototype architecture                 ✓
```

The next stage is to transfer this experimental knowledge to the **actual contest dataset**.

---

# 38. Files

The important AI4I experiment files are:

```text
paper_cswe.py
```

Main paper-inspired architecture.

```text
experiment1_multiseed.py
```

Robustness evaluation across multiple random seeds.

Additional result files may include:

```text
custom_cswe_results.csv
normalized_cswe_results.csv
experiment1_multiseed_results.csv
```

These contain the numerical results produced by the experiments.

---

# 39. Final Takeaway

The AI4I work should be viewed as our **proof-of-concept stage**.

We started with a highly imbalanced predictive-maintenance dataset and progressively built:

```text
Sensor features
      ↓
Sparse Autoencoder
      ↓
Hybrid representation
      ↓
Multiple classifiers
      ↓
Failure competence
      ↓
Weighted ensemble
      ↓
Confidence-aware threshold
      ↓
Robust multi-seed evaluation
```

The experiments also taught us when **not** to use a component.

The paper-style CSWE was initially worse than Bagging.

After modifying the failure weighting and decision threshold, performance improved substantially.

The multi-seed experiment then showed that the improved system was not simply the result of one lucky split.

Therefore, the main output of the AI4I stage is not just a set of accuracy numbers.

It is a **working predictive-maintenance experimentation framework** that we can now take to the current contest dataset, where the real objective is to discover which combination of features, representation learning, classifiers, ensemble weighting, and decision thresholds actually works best.
