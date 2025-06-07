# Neurobehavioral Risk Detection in Online Gambling

**A Data-Driven Approach for Early Intervention Using Neuroscience-Informed Machine Learning**

## Abstract

This repository contains a neuroscience-informed machine learning approach for early detection of gambling addiction risk. The methodology combines synthetic behavioral data generation with brain-based feature engineering to achieve 84.5% classification accuracy and 98% ROC AUC for high-risk detection. The approach provides a privacy-preserving framework for gambling addiction research while maintaining clinical interpretability through neuroscience grounding.

## Key Results

- **84.5% classification accuracy** across three risk categories (Normal, At-Risk, High-Risk)
- **98% ROC AUC** for high-risk gambling addiction detection
- **Large effect sizes** (η² > 0.39) for primary behavioral features
- **Neuroscience validation** mapping features to specific brain mechanisms
- **Clinical viability** with flexible confidence thresholds for intervention

## Neuroscience-Informed Features

Our approach maps gambling behaviors to established addiction neuroscience:

| Feature | Brain Region | Mechanism | Literature |
|---------|--------------|-----------|------------|
| **Session Intensity** | Anterior Cingulate Cortex | Compulsive monitoring | Potenza et al., 2003 |
| **Average Bet Size** | Prefrontal Cortex | Impulse control | Tanabe et al., 2007 |
| **Loss Chasing Score** | Ventral Striatum | Reward prediction error | Reuter et al., 2005 |

## Quick Start

### Prerequisites

**Install Dependencies:**
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn shap joblib
```

### Running the Complete Analysis Pipeline

**Step 1: Generate Synthetic Gambling Data**
```bash
python gambling_simulator.py
```
**Output:** Creates `gambling_features.csv` and `gambling_sessions.csv` with behavioral data for 1000 users

**Step 2: Train the Risk Detection Model**
```bash
python realistic_model.py
```
**Output:** 
- `gambling_features_very_realistic.csv` (final dataset)
- `gambling_risk_model_final.pkl` (trained model)
- `feature_scaler.pkl` (preprocessing parameters)
- Complete performance metrics printed to console

**Step 3: Statistical Validation**
```bash
python statistical_tests.py
```
**Output:** ANOVA results, effect sizes, and statistical significance testing

**Step 4: Model Interpretability Analysis**
```bash
python interpretability_analysis.py
```
**Output:** SHAP analysis, ROC curves, and clinical decision thresholds

### Expected Runtime
- **Total execution time:** 5-10 minutes
- **Data generation:** 2-3 minutes
- **Model training:** 1-2 minutes
- **Statistical tests:** 1 minute
- **Interpretability analysis:** 2-3 minutes

## Detailed Usage Examples

### Loading and Using the Trained Model

```python
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('gambling_risk_model_final.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Load sample data
data = pd.read_csv('gambling_features_very_realistic.csv')

# Prepare features for prediction
feature_columns = ['loss_chasing_score', 'session_intensity', 'avg_bet_size', 
                  'max_loss_streak', 'total_bets', 'session_duration_min']

# Make predictions for new users
X = data[feature_columns].head(5)  # First 5 users
predictions = model.predict(X)
probabilities = model.predict_proba(X)

print("Risk Predictions:", predictions)
print("Risk Probabilities:", probabilities)
```

### Generating Custom Synthetic Data

```python
from gambling_simulator import GamblingBehaviorSimulator

# Create simulator with custom parameters
simulator = GamblingBehaviorSimulator(seed=123)

# Generate custom user session
custom_session = simulator.generate_user_session(
    user_type='high_risk', 
    session_length_minutes=120
)

print(f"Generated {len(custom_session)} bets for high-risk user")
```

### Analyzing Individual Risk Factors

```python
import shap

# Load model and create explainer
model = joblib.load('gambling_risk_model_final.pkl')
explainer = shap.TreeExplainer(model)

# Explain predictions for specific users
X_sample = data[feature_columns].head(1)
shap_values = explainer.shap_values(X_sample)

print("Feature contributions to risk prediction:")
for feature, contribution in zip(feature_columns, shap_values[0]):
    print(f"{feature}: {contribution:.3f}")
```

## Results Summary

### Classification Performance

```
Overall Accuracy: 84.5%
Cross-Validation: 84.2% ± 9.3%

Per-Class Performance:
- Normal Users:    Precision=87.8%, Recall=93.1%, F1=90.4%
- At-Risk Users:   Precision=76.9%, Recall=69.0%, F1=72.7%  
- High-Risk Users: Precision=77.8%, Recall=63.6%, F1=70.0%

ROC AUC Scores:
- Normal: 0.939, At-Risk: 0.923, High-Risk: 0.979
```

### Feature Importance Hierarchy

1. **Session Intensity** (31.8%) - Compulsive behavior patterns
2. **Total Bets** (28.3%) - Behavioral frequency
3. **Average Bet Size** (18.7%) - Impulse control deficits
4. **Session Duration** (12.0%) - Temporal regulation
5. **Maximum Loss Streak** (5.0%) - Emotional regulation
6. **Loss Chasing Score** (4.2%) - Reward system dysfunction

### Statistical Validation

All behavioral features showed significant differences between risk groups (p < 0.001):

- **Session Intensity**: F(2,997) = 401.17, η² = 0.446 (Large effect)
- **Total Bets**: F(2,997) = 401.18, η² = 0.446 (Large effect)
- **Average Bet Size**: F(2,997) = 321.18, η² = 0.392 (Large effect)

## Clinical Applications

### Decision Support Thresholds

The model provides flexible deployment options for different clinical scenarios:

- **Sensitive Screening (30%)**: 91.1% recall, 72.9% precision
- **Balanced Detection (50%)**: 83.9% recall, 85.5% precision
- **Conservative Intervention (70%)**: 73.2% recall, 91.1% precision
- **High-Confidence Referral (90%)**: 51.8% recall, 90.6% precision

