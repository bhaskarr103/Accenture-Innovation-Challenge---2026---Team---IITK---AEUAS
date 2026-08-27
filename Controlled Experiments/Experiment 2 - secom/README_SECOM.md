# Predictive Maintenance — SECOM Experiments

## Overview

After completing the initial AI4I 2020 prototype, we moved to the **SECOM semiconductor manufacturing dataset**.

The purpose of the SECOM stage was not simply to obtain a high score.

The goal was to test whether the architecture developed on AI4I could generalize to a substantially more difficult industrial dataset with:

- very high dimensionality,
- missing sensor measurements,
- duplicate measurements,
- many irrelevant/redundant variables,
- stronger feature-space complexity,
- and an imbalanced failure/success target.

The SECOM experiments therefore became our second experimental stage.

The main question was:

> Can the Sparse Autoencoder + classifier ensemble + competence/threshold methodology developed on AI4I be transferred to a much higher-dimensional industrial process dataset?

The experiments showed that the answer is **not directly**.

The SECOM results exposed several important weaknesses in the current architecture and gave us a much clearer direction for the next stage.

---

# 1. Dataset — SECOM

## 1.1 What is SECOM?

SECOM is a semiconductor manufacturing dataset containing measurements collected from a manufacturing process.

The dataset is substantially different from AI4I.

AI4I contained:

```text
10,000 samples
14 columns
```

while SECOM contains:

```text
1,567 samples
590 measurement features
```

The raw feature matrix was:

```text
(1567, 590)
```

The dataset therefore has a very different dimensionality regime:

```text
AI4I:
~10,000 samples
~5–8 useful input dimensions

SECOM:
1,567 samples
590 raw measurement dimensions
```

This immediately makes SECOM a much more difficult machine-learning problem.

---

# 2. SECOM Files

The SECOM dataset was provided through two main files:

```text
secom.data
```

containing the process measurements,

and:

```text
secom_labels.data
```

containing the target labels and timestamps.

The labels file had:

```text
1567 rows
2 columns
```

The first column contained the manufacturing outcome.

The second column contained the timestamp.

---

# 3. Target Distribution

The SECOM labels were:

```text
-1 = Normal
 1 = Failure
```

The distribution was:

```text
Normal  = 1463
Failure = 104
```

Therefore:

```text
Normal  = 93.36%
Failure =  6.64%
```

This is still an imbalanced classification problem.

However, the imbalance is less extreme than AI4I.

AI4I:

```text
Failure = 3.39%
```

SECOM:

```text
Failure = 6.64%
```

Therefore, SECOM gives us slightly more positive examples, but the absolute number is still very small:

```text
104 failures
```

out of:

```text
1567 samples
```

---

# 4. Why SECOM Is More Difficult

SECOM immediately introduced several problems that were not as severe in AI4I.

The raw dataset contained:

```text
590 features
```

but:

```text
41951 missing values
```

were present.

There were:

```text
538 columns containing missing values
```

The missing-value percentage ranged from:

```text
0%
```

to:

```text
91.19%
```

with an average of approximately:

```text
4.54%
```

This means that the raw SECOM matrix cannot simply be passed directly into a classifier.

---

# 5. Duplicate Features

Another important discovery was that SECOM contained duplicate feature columns.

We identified:

```text
104 duplicate columns
```

Therefore:

```text
590 raw features
        ↓
remove duplicate columns
        ↓
486 features
```

This was our first major feature-cleaning step.

Duplicate features do not provide additional information and can unnecessarily increase dimensionality.

---

# 6. Missing-Value Feature Filtering

After duplicate removal, we investigated the amount of missing data in each feature.

We removed features with more than:

```text
50%
```

missing values.

This removed:

```text
28 features
```

Therefore:

```text
486
 ↓
remove high-missing features
 ↓
458 final features
```

The final feature dimension became:

```text
458
```

So the preprocessing pipeline was:

```text
Raw SECOM
590 features
      ↓
Duplicate removal
486 features
      ↓
Remove >50% missing
458 features
```

---

# 7. Train / Validation / Test Split

The dataset was split into:

```text
Training   : 1096 samples
Validation : 235 samples
Test       : 236 samples
```

The class distributions were approximately:

```text
Training:
Normal  = 1023
Failure =   73

Validation:
Normal  = 220
Failure =  15

Test:
Normal  = 220
Failure =  16
```

Therefore:

```text
Training failure rate   = 6.66%
Validation failure rate = 6.38%
Test failure rate       = 6.78%
```

The split was kept stratified so that the failure proportion remained similar across the three subsets.

---

# 8. Missing-Value Imputation

After feature filtering, the remaining missing values were handled using median imputation.

The procedure was:

```text
Training data
      ↓
Calculate feature medians
      ↓
Replace missing values
      ↓
Apply same learned medians
to validation/test
```

After imputation:

```text
Training NaNs   = 0
Validation NaNs = 0
Test NaNs       = 0
```

This gave us a complete numerical matrix suitable for the SAE and classifiers.

---

# 9. Standardization

The 458 remaining features were standardized using the training data.

Conceptually:

```text
x_standardized =
(x - training_mean) / training_std
```

The transformation was fitted only using the training set and then applied to validation and test data.

After standardization, the training features had approximately:

```text
Mean = 0
Std  = 1
```

for non-zero-variance features.

This was important because SECOM contains measurements with dramatically different numerical scales.

---

# 10. Final SECOM Input

The preprocessing pipeline therefore became:

```text
SECOM
  ↓
590 raw measurements
  ↓
Remove 104 duplicate columns
  ↓
486 features
  ↓
Remove 28 features with >50% missing
  ↓
458 features
  ↓
Median imputation
  ↓
Standardization
  ↓
458-dimensional input
```

This became the base input for the SAE and classifier experiments.

---

# 11. Initial SECOM Sparse Autoencoder

We then attempted to transfer the SAE architecture from AI4I to SECOM.

However, SECOM required a much larger encoder because the input dimension was:

```text
458
```

instead of:

```text
8
```

The initial architecture was:

```text
Input
458
 ↓
Hidden 1
128
 ↓
Hidden 2
32
 ↓
Latent
8
```

Therefore:

```text
458 → 128 → 32 → 8
```

The latent representation was then decoded back toward the original 458-dimensional space.

---

# 12. SAE Configuration

The SAE used:

```text
Input dimension   : 458
Hidden dimension 1: 128
Hidden dimension 2: 32
Latent dimension  : 8
```

Sparsity target:

```text
rho = 0.05
```

The initial sparsity weight was:

```text
beta = 0.1
```

with L2 regularization:

```text
1e-5
```

The training objective remained conceptually:

```text
Total Loss
=
Reconstruction Loss
+
Sparsity Penalty
+
L2 Regularization
```

---

# 13. First SAE Problem — Latent Collapse

The first SAE configuration produced:

```text
Mean latent activation:

[0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05]
```

and approximately zero variation in the latent representation.

This was a serious problem.

The SAE was technically satisfying the sparsity target, but it was doing so by producing nearly constant latent representations.

Conceptually:

```text
Different SECOM samples
        ↓
almost identical latent vectors
        ↓
little useful information
```

This is not useful representation learning.

---

# 14. Fixing SAE Latent Collapse

We reduced the sparsity weight:

```text
beta:
0.1 → 0.01
```

This allowed reconstruction to have more influence while still retaining the sparsity constraint.

The resulting latent activations became:

```text
[0.0341
 0.2864
 0.0879
 0.0615
 0.0669
 0.0672
 0.0582
 0.0963]
```

More importantly, the latent standard deviations became:

```text
[0.0725
 0.0433
 0.1047
 0.1282
 0.1504
 0.1220
 0.0931
 0.1308]
```

Therefore:

```text
Collapsed neurons = 0 / 8
```

The SAE was no longer obviously collapsed.

This was an important result because it demonstrated that the SAE hyperparameters must be adapted to the dataset.

---

# 15. Latent Feature Extraction

After training the SAE, we extracted the 8-dimensional latent representation.

The transformation was:

```text
458 original features
        ↓
       SAE
        ↓
8 latent features
```

The resulting datasets were:

```text
Training   : (1096, 8)
Validation : (235, 8)
Test       : (236, 8)
```

The latent representation was saved as:

```text
secom_sae_features.npz
```

This allowed the SAE and classifier experiments to be separated.

---

# 16. Hybrid Representation

Following the AI4I approach, we also constructed a hybrid representation:

```text
Original 458 features
+
SAE latent 8 features
```

Therefore:

```text
458 + 8 = 466 features
```

The three representations we eventually compared were:

```text
Original:
458 features

SAE:
8 features

Hybrid:
466 features
```

This became the basis for our SAE ablation experiments.

---

# 17. Classifier Bank

We transferred the same classifier-bank idea from AI4I.

The classifiers were:

```text
SVM
KNN
GaussianNB
Bagging
```

The goal was to determine whether different classifiers would specialize differently on SECOM failures.

---

# 18. Initial SECOM Classifier Results

The validation results immediately revealed a major difference from AI4I.

Using the hybrid 466-dimensional representation:

```text
Classifier        Normal F1   Failure F1   Failure Recall
----------------------------------------------------------
SVM                  0.9623       0.1053       0.0667
KNN                  0.9648       0.0000       0.0000
GaussianNB           0.3582       0.1485       1.0000
Bagging              0.9670       0.0000       0.0000
```

This was very revealing.

Unlike AI4I:

```text
Bagging
```

was no longer automatically strong at detecting failures.

Instead:

```text
GaussianNB
```

detected:

```text
100%
```

of validation failures, but with extremely poor precision.

This showed that the classifier behaviours were radically different on SECOM.

---

# 19. Failure Competence

The resulting failure-specific competence weights were:

```text
SVM          0.4148
KNN          0.0000
GaussianNB   0.5852
Bagging      0.0000
```

This is an interesting result.

On AI4I:

```text
Bagging
```

was the dominant failure classifier.

On SECOM:

```text
GaussianNB
```

became the strongest failure-recall classifier.

This demonstrates why fixed ensemble assumptions are dangerous.

The classifier that is best for one industrial dataset is not necessarily best for another.

---

# 20. Initial SECOM Ensemble Results

The initial ensemble experiment produced:

```text
Bagging
Accuracy  : 93.22%
Precision : 0.00%
Recall    : 0.00%
F1        : 0.00%
```

Confusion matrix:

```text
[[220   0]
 [ 16   0]]
```

Bagging predicted:

```text
0 failures
```

and detected:

```text
0 / 16
```

actual failures.

---

# 21. Majority Voting

Majority voting performed slightly differently:

```text
Accuracy  : 92.80%
Precision : 40.00%
Recall    : 12.50%
F1        : 19.05%
```

Confusion matrix:

```text
[[217   3]
 [ 14   2]]
```

It predicted:

```text
5 failures
```

and detected:

```text
2 / 16
```

actual failures.

Therefore, simple majority voting was also not sufficient.

---

# 22. Original CSWE on SECOM

The original competence-weighted CSWE also struggled.

Its result was:

```text
Accuracy  : 93.22%
Precision : 0.00%
Recall    : 0.00%
F1        : 0.00%
```

It predicted:

```text
0 failures
```

and detected:

```text
0 / 16
```

This was a strong warning.

The AI4I architecture could not simply be transferred to SECOM without adaptation.

---

# 23. Why Threshold Learning Was Necessary

The probability outputs showed another problem.

Using the default classification threshold:

```text
0.50
```

many models almost never predicted the failure class.

For example, Bagging had:

```text
Validation maximum probability ≈ 0.30
Test maximum probability ≈ 0.41
```

Therefore, a threshold of:

```text
0.50
```

would automatically produce almost no failure predictions.

This motivated a dedicated threshold-learning experiment.

---

# 24. Experiment — Failure Threshold Learning

We trained the classifiers and inspected their validation probability distributions.

The best thresholds were approximately:

```text
SVM:
0.10

KNN:
0.15

GaussianNB:
0.17

Bagging:
0.14
```

These were dramatically lower than the default:

```text
0.50
```

This demonstrated that probability threshold selection is extremely important on SECOM.

---

# 25. Threshold Results

The learned-threshold test results were:

```text
SVM
Accuracy  = 86.86%
Precision = 14.29%
Recall    = 18.75%
F1        = 16.22%

KNN
Accuracy  = 89.83%
Precision = 21.43%
Recall    = 18.75%
F1        = 20.00%

GaussianNB
Accuracy  = 18.64%
Precision = 6.86%
Recall    = 87.50%
F1        = 12.73%

Bagging
Accuracy  = 85.17%
Precision = 21.21%
Recall    = 43.75%
F1        = 28.57%
```

This showed the trade-off very clearly.

For example:

```text
GaussianNB
```

could detect almost all failures, but at the cost of predicting almost everything as failure.

Bagging provided a more balanced operating point.

---

# 26. Threshold Trade-Off

The SECOM experiments showed that there is no universally correct threshold.

For example:

```text
Lower threshold
      ↓
More predicted failures
      ↓
Higher recall
      ↓
Lower precision
```

while:

```text
Higher threshold
      ↓
Fewer predicted failures
      ↓
Higher precision
      ↓
Lower recall
```

Therefore, the threshold has to be selected according to the competition's evaluation metric and the cost of false negatives versus false positives.

---

# 27. Calibration Experiment

We then attempted probability calibration.

The purpose was to determine whether the raw classifier probabilities could be made more comparable across classifiers.

We calibrated:

```text
SVM
KNN
GaussianNB
Bagging
```

However, the calibrated probabilities became highly compressed around the class prior.

For example, the calibrated means were approximately:

```text
SVM          ≈ 0.0665
KNN          ≈ 0.0667
GaussianNB   ≈ 0.0667
Bagging      ≈ 0.0654
```

This did not produce a useful improvement in the final CSWE system.

The calibrated confidence-aware ensemble predicted:

```text
0 failures
```

on the test set.

Therefore:

```text
Calibration experiment
        ↓
Did not improve the current architecture
```

This is an important negative result.

---

# 28. Experiment 2D — SAE Ablation

We then asked an important question:

> Is the SAE actually helping SECOM?

To answer this, we compared:

```text
Original 458 features
SAE 8 features
Hybrid 466 features
```

using the same Bagging + threshold-learning methodology.

---

# 29. SAE Ablation Results

The results were:

```text
Feature Set      Accuracy   Precision   Recall    F1
---------------------------------------------------------
Original 458       90.68%     33.33%    37.50%   35.29%
SAE 8              61.86%      7.95%    43.75%   13.46%
Hybrid 466         85.17%     21.21%    43.75%   28.57%
```

The important result was:

```text
Original 458
```

performed better than the SAE-only and hybrid configurations.

---

# 30. Interpretation of Experiment 2D

This result does **not** mean:

> "SAEs are useless."

It means:

> The current SAE representation and configuration did not improve the SECOM classification problem.

The original features contained important information that was being lost or distorted by the compressed representation.

The hybrid representation recovered some of that information, but still did not outperform the original feature space.

This is exactly why ablation experiments are important.

---

# 31. Experiment 2E — Latent Dimension Ablation

We then tested whether the problem was simply the latent dimension.

We compared:

```text
Latent = 8
Latent = 16
Latent = 32
```

For each latent size we evaluated:

```text
SAE only
```

and:

```text
Original + SAE hybrid
```

---

# 32. Latent-8 Results

For:

```text
Latent dimension = 8
```

we obtained:

```text
SAE-8

Accuracy  : 66.10%
Precision : 7.89%
Recall    : 37.50%
F1        : 13.04%
```

Hybrid:

```text
Hybrid-8

Accuracy  : 83.90%
Precision : 19.44%
Recall    : 43.75%
F1        : 26.92%
```

---

# 33. Latent-16 Results

For:

```text
Latent dimension = 16
```

SAE-only:

```text
Accuracy  : 85.17%
Precision : 12.00%
Recall    : 18.75%
F1        : 14.63%
```

Hybrid:

```text
Accuracy  : 88.14%
Precision : 20.00%
Recall    : 25.00%
F1        : 22.22%
```

---

# 34. Latent-32 Results

For:

```text
Latent dimension = 32
```

SAE-only:

```text
Accuracy  : 79.66%
Precision : 10.00%
Recall    : 25.00%
F1        : 14.29%
```

Hybrid:

```text
Accuracy  : 86.44%
Precision : 21.43%
Recall    : 37.50%
F1        : 27.27%
```

---

# 35. Complete Latent Dimension Ablation

The complete experiment was:

```text
Method          Dimension   Accuracy   Precision   Recall    F1
------------------------------------------------------------------
Original 458       458        90.68%     33.33%    37.50%   35.29%

SAE-8                8        66.10%      7.89%    37.50%   13.04%
Hybrid-8           466        83.90%     19.44%    43.75%   26.92%

SAE-16              16        85.17%     12.00%    18.75%   14.63%
Hybrid-16          474        88.14%     20.00%    25.00%   22.22%

SAE-32              32        79.66%     10.00%    25.00%   14.29%
Hybrid-32          490        86.44%     21.43%    37.50%   27.27%
```

The best result in this experiment was:

```text
Original 458
```

with:

```text
F1 = 35.29%
Recall = 37.50%
Precision = 33.33%
```

---

# 36. What Experiment 2E Taught Us

This was one of the most important SECOM experiments.

Increasing the latent dimension did not automatically improve performance.

We tested:

```text
8
16
32
```

but none of the hybrid configurations surpassed the original 458-feature representation.

Therefore:

```text
More latent dimensions
        ≠
Better representation
```

and:

```text
SAE compression
        ≠
Guaranteed classification improvement
```

The raw feature space still contained the strongest predictive signal under our current methodology.

---

# 37. SECOM vs AI4I

The contrast between the two datasets is extremely useful.

## AI4I

```text
10,000 samples
~5–8 useful input dimensions
3.39% failure
Relatively low dimensional
```

Our prototype achieved strong performance:

```text
Bagging:
F1 ≈ 74.74% mean
Recall ≈ 67.84% mean
```

## SECOM

```text
1,567 samples
458 cleaned features
6.64% failure
Very high dimensional
Many missing values
104 duplicate columns
```

The best simple thresholded Bagging result was:

```text
F1 = 35.29%
Recall = 37.50%
```

This demonstrates that the same architecture behaves very differently depending on the data structure.

---

# 38. Why the AI4I Architecture Did Not Transfer Directly

There are several likely reasons visible from our experiments.

## 38.1 Dimensionality

AI4I:

```text
8 input dimensions
```

SECOM:

```text
458 input dimensions
```

The ratio between features and samples is dramatically different.

SECOM has:

```text
458 features
1096 training samples
```

which makes the learning problem much harder.

---

## 38.2 Missing Data

SECOM contains extensive missing measurements.

Even after filtering:

```text
458 features
```

still required imputation.

This can distort the learned feature distribution.

---

## 38.3 Redundant Measurements

SECOM originally contained:

```text
104 duplicate columns
```

This demonstrates that the raw feature space contains substantial redundancy.

---

## 38.4 Small Number of Failures

There are only:

```text
104 total failures
```

and only:

```text
73 training failures
15 validation failures
16 test failures
```

This means every validation/test failure has a large effect on the measured score.

For example, on the test set:

```text
1 correctly detected failure
```

changes recall by:

```text
1 / 16 = 6.25 percentage points
```

Therefore, SECOM metrics naturally have high variance.

---

# 39. Why Multi-Seed Validation Becomes Even More Important

On AI4I:

```text
51 test failures
```

were available in each split.

On SECOM:

```text
16 test failures
```

were available in our split.

Therefore, a single SECOM test result is much less stable.

For the final competition pipeline, we should therefore avoid making architectural decisions based on one split whenever possible.

A future experiment should use:

```text
Stratified K-fold cross-validation
```

or repeated stratified splits.

---

# 40. Important Negative Results

The SECOM stage produced several useful negative results.

### Negative Result 1

The original CSWE did not outperform Bagging.

### Negative Result 2

Raw 0.50 probability thresholds were inappropriate.

### Negative Result 3

Threshold optimization improved failure detection but reduced precision.

### Negative Result 4

Probability calibration did not solve the ensemble problem.

### Negative Result 5

The SAE-only representation was significantly weaker than the original feature space.

### Negative Result 6

Hybrid SAE features did not outperform the original features.

### Negative Result 7

Increasing latent dimension from:

```text
8 → 16 → 32
```

did not solve the problem.

These are not wasted experiments.

They tell us which directions should not be blindly pursued.

---

# 41. What We Learned About the SAE

The SECOM experiments changed our understanding of the Sparse Autoencoder.

The original assumption was:

```text
High-dimensional data
        ↓
SAE compression
        ↓
Better representation
        ↓
Better classifier
```

But our experiments showed:

```text
High-dimensional data
        ↓
SAE compression
        ↓
Potential information loss
        ↓
Worse classifier performance
```

Therefore, for the contest we should treat SAE as an experimental component rather than a mandatory component.

The correct question is:

> Does the learned representation improve validation performance?

If not, the original features should be retained.

---

# 42. What We Learned About Ensemble Design

AI4I taught us:

```text
Competence-aware weighting can help.
```

SECOM taught us:

```text
Competence-aware weighting can also fail.
```

Why?

Because classifier behaviour is dataset-dependent.

For AI4I:

```text
Bagging
```

was strongest.

For SECOM:

```text
GaussianNB
```

had extremely high failure recall but terrible precision.

Therefore, the ensemble should not simply ask:

```text
Which classifier has the highest failure recall?
```

It needs to consider:

```text
Failure F1
Failure recall
Precision
Probability quality
Decision threshold
Stability
```

---

# 43. Better Direction Suggested by SECOM

The SECOM experiments suggest that our next ensemble design should be more adaptive.

Instead of:

```text
Static classifier weights
```

we should investigate:

```text
Validation-based dynamic weights
```

potentially based on:

```text
Failure F1
Failure recall
Precision
Confidence
Local/sample-level competence
```

This would be closer to the underlying motivation of competence-based ensemble learning.

---

# 44. Potential Next Architecture

The lessons from AI4I + SECOM suggest a more general architecture:

```text
                    RAW DATA
                       │
                       ▼
                DATA CLEANING
                       │
                       ▼
             FEATURE ENGINEERING
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
    ORIGINAL FEATURES        LEARNED FEATURES
          │                         │
          │                    SAE / Other
          │                         │
          └────────────┬────────────┘
                       ▼
                 MODEL BANK
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         SVM          KNN          GNB
                       │
                       ▼
                    Bagging
                       │
                       ▼
              VALIDATION ANALYSIS
                       │
                       ▼
             FAILURE COMPETENCE
                       │
                       ▼
             CONFIDENCE / THRESHOLD
                       │
                       ▼
                FINAL DECISION
```

But importantly, every block should be optional.

---

# 45. Proposed Feature-Selection Stage

SECOM strongly suggests that feature selection should become a major component of the next system.

Instead of immediately feeding:

```text
458 features
```

into the model, we should investigate:

```text
458
 ↓
Variance filtering
 ↓
Correlation filtering
 ↓
Feature importance
 ↓
Mutual information
 ↓
Model-based selection
 ↓
Top-K features
```

For example:

```text
Top 20
Top 50
Top 100
Top 200
All 458
```

can be compared.

This may be more useful than simply increasing SAE latent dimension.

---

# 46. Proposed Contest Strategy

The SECOM results suggest the following experimental order for the current contest dataset.

## Step 1 — Strong baseline

Train:

```text
Logistic Regression
Random Forest / Bagging
SVM
```

without SAE.

---

## Step 2 — Feature selection

Test:

```text
Top-K features
```

with multiple K values.

---

## Step 3 — Imbalance handling

Test:

```text
Class weights
Threshold tuning
Possibly resampling
```

while avoiding leakage.

---

## Step 4 — Classifier bank

Train:

```text
SVM
KNN
Naive Bayes
Bagging
Random Forest
Gradient boosting
```

depending on the contest data.

---

## Step 5 — Competence analysis

Measure:

```text
Failure precision
Failure recall
Failure F1
```

for every classifier.

---

## Step 6 — Ensemble

Compare:

```text
Best single classifier
Majority voting
Weighted voting
Confidence-aware fusion
```

---

## Step 7 — SAE

Only after establishing the baseline:

```text
Train SAE
Extract latent features
Compare against raw features
```

---

## Step 8 — Robustness

Run:

```text
Multiple seeds
or
Stratified K-fold
```

before selecting the final architecture.

---

# 47. SECOM Experiment Files

The SECOM experiments were divided into several stages.

The preprocessing stage established:

```text
590
 ↓
486
 ↓
458
```

features.

The SAE experiment established:

```text
458
 ↓
8 latent
```

features.

The ensemble experiment tested:

```text
SVM
KNN
GaussianNB
Bagging
```

The threshold experiment investigated:

```text
Default threshold
vs
Validation-learned threshold
```

The ablation experiments investigated:

```text
Original
vs
SAE
vs
Hybrid
```

and finally:

```text
Latent 8
Latent 16
Latent 32
```

---

# 48. Experiment 2D

Experiment 2D was the:

```text
SAE ABLATION EXPERIMENT
```

It compared:

```text
Original 458
SAE 8
Hybrid 466
```

Main conclusion:

```text
Original 458 was best.
```

---

# 49. Experiment 2E

Experiment 2E was the:

```text
LATENT DIMENSION ABLATION
```

It compared:

```text
8 latent neurons
16 latent neurons
32 latent neurons
```

for both:

```text
SAE-only
```

and:

```text
Hybrid
```

Main conclusion:

```text
Increasing latent dimension did not solve the problem.
```

The best overall result remained:

```text
Original 458
```

with:

```text
F1 = 35.29%
Recall = 37.50%
Precision = 33.33%
```

---

# 50. Current SECOM Status

The SECOM prototype can be summarized as:

```text
SECOM
 │
 ├── Dataset inspection                         ✓
 │
 ├── Missing-value analysis                     ✓
 │
 ├── Duplicate-feature analysis                 ✓
 │
 ├── Feature filtering                          ✓
 │
 ├── Median imputation                          ✓
 │
 ├── Standardization                            ✓
 │
 ├── Sparse Autoencoder                         ✓
 │
 ├── Latent-collapse diagnosis                  ✓
 │
 ├── Latent dimension experiments               ✓
 │
 ├── Hybrid feature experiments                 ✓
 │
 ├── Multiple classifiers                       ✓
 │
 ├── Competence weighting                       ✓
 │
 ├── Threshold learning                         ✓
 │
 ├── Probability calibration                    ✓
 │
 ├── SAE ablation                               ✓
 │
 └── Latent dimension ablation                  ✓
```

---

# 51. Final SECOM Conclusion

The SECOM experiments were valuable because they challenged the assumptions made during the AI4I stage.

The main conclusion is:

> The AI4I architecture cannot simply be transferred to a high-dimensional industrial dataset and expected to work unchanged.

SECOM demonstrated that:

```text
High dimensionality
+
Missing values
+
Feature redundancy
+
Small sample size
+
Class imbalance
```

create a substantially different learning problem.

The original 458-feature representation actually outperformed our current SAE and hybrid representations.

The best result from the SAE ablation was:

```text
Original 458
Accuracy  : 90.68%
Precision : 33.33%
Recall    : 37.50%
F1        : 35.29%
```

The best hybrid configuration reached:

```text
Hybrid-8
Recall = 43.75%
F1     = 26.92%
```

and:

```text
Hybrid-32
Recall = 37.50%
F1     = 27.27%
```

but neither surpassed the original feature representation in F1.

---

# 52. The Most Important Lesson

The biggest lesson from SECOM is:

> **Do not force the architecture to fit the dataset. Make the architecture adapt to the dataset.**

On AI4I:

```text
SAE + ensemble + confidence
```

was promising.

On SECOM:

```text
Raw features + thresholded classifier
```

performed better than our current SAE-based representation.

Therefore, the final contest architecture should be selected empirically.

We should not decide beforehand that:

```text
"SAE must be used."
```

or:

```text
"CSWE must be used."
```

Instead:

```text
Dataset
   ↓
Experiments
   ↓
Evidence
   ↓
Best architecture
```

---

# 53. Connection to the Current Contest

The SECOM experiment is particularly useful for the current competition because it has exposed the exact type of problems we need to be prepared for.

The contest dataset may contain:

```text
Many features
Missing values
Redundant features
Imbalanced classes
Limited failure examples
```

and we now have experience handling all of these.

We have already developed code and experimental procedures for:

```text
Data inspection
Missing-value handling
Duplicate detection
Feature filtering
Imputation
Standardization
SAE training
Latent extraction
Classifier banks
Competence estimation
Threshold optimization
Ablation studies
Multi-model comparison
```

Therefore, the contest stage can begin from a much stronger foundation.

---

# 54. AI4I → SECOM → Contest

The project progression is now:

```text
                    AI4I
                     │
                     ▼
          Understand the architecture
                     │
                     ▼
          Build paper-style CSWE
                     │
                     ▼
       Improve weighting + threshold
                     │
                     ▼
             Multi-seed validation
                     │
                     ▼
                  SECOM
                     │
                     ▼
        Test generalization to
       high-dimensional industrial data
                     │
                     ▼
         Discover SAE limitations
                     │
                     ▼
        Discover threshold importance
                     │
                     ▼
          Perform SAE ablations
                     │
                     ▼
              Current Contest
                     │
                     ▼
        Build dataset-specific system
```

This is the progression of the project.

---

# 55. Final Project Philosophy

The project is no longer simply an attempt to reproduce a paper.

The paper provided the initial idea:

```text
Sparse representation
+
Multiple classifiers
+
Classifier competence
```

AI4I allowed us to build and improve the concept.

SECOM allowed us to stress-test it.

The next stage is to use those lessons to build a **dataset-adaptive predictive-maintenance system for the actual contest**.

The ultimate objective is therefore:

```text
Not:
"Use the paper exactly."

But:

"Understand the paper,
build the idea,
test it,
find its limitations,
improve it,
and select what actually works
for the target dataset."
```

---

# 56. Final Takeaway

The SECOM stage gave us three major pieces of knowledge:

### 1. Raw features are extremely important

The 458-dimensional original representation outperformed our current SAE representations.

### 2. Thresholds matter

The default 0.50 probability threshold was clearly inappropriate for several SECOM classifiers.

### 3. Architecture must be validated

The same SAE + CSWE architecture that worked reasonably well on AI4I did not automatically work on SECOM.

This is exactly why the SECOM experiments were performed.

They have turned the project from:

```text
Paper implementation
```

into:

```text
Experimental predictive-maintenance framework
```

that can now be adapted to the actual competition problem.
