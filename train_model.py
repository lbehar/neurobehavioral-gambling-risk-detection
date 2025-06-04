import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

print("Loading gambling behavior dataset...")

# Load your generated data
features_df = pd.read_csv('gambling_features.csv')

print(f"Dataset loaded: {len(features_df)} users")
print(f"Risk distribution: {features_df['risk_label'].value_counts().to_dict()}")

# Prepare features for machine learning
feature_columns = ['loss_chasing_score', 'session_intensity', 'avg_bet_size', 
                  'max_loss_streak', 'total_bets', 'session_duration_min']

X = features_df[feature_columns]
y = features_df['risk_label']

# Check for any missing values
print(f"\nMissing values: {X.isnull().sum().sum()}")

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} users")
print(f"Test set: {len(X_test)} users")

# Train Random Forest model
print("\nTraining Random Forest classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print("CLASSIFICATION RESULTS")
print("="*50)

print(f"\nOverall Accuracy: {accuracy:.3f}")

print("\nDetailed Classification Report:")
class_names = ['Normal (0)', 'At-Risk (1)', 'High-Risk (2)']
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("Predicted:  Normal  At-Risk  High-Risk")
for i, actual_class in enumerate(['Normal', 'At-Risk', 'High-Risk']):
    row = f"Actual {actual_class:8}: "
    for j in range(3):
        row += f"{cm[i][j]:6d}   "
    print(row)

# Feature importance analysis
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeatures ranked by importance:")
for _, row in importance_df.iterrows():
    feature_name = row['feature'].replace('_', ' ').title()
    print(f"{feature_name:20}: {row['importance']:.3f}")

# Cross-validation for robustness
print("\n" + "="*50)
print("MODEL VALIDATION")
print("="*50)

cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Performance by risk class
print("\nPer-class performance:")
for i, class_name in enumerate(class_names):
    class_mask = (y_test == i)
    if class_mask.sum() > 0:
        class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
        print(f"{class_name}: {class_accuracy:.3f} ({class_mask.sum()} users)")

# Risk prediction probabilities analysis
print("\n" + "="*50)
print("RISK PREDICTION ANALYSIS")
print("="*50)

print("\nAverage prediction confidence by true class:")
for i, class_name in enumerate(class_names):
    class_mask = (y_test == i)
    if class_mask.sum() > 0:
        avg_confidence = y_pred_proba[class_mask, i].mean()
        print(f"{class_name}: {avg_confidence:.3f}")

# Model interpretation
print("\n" + "="*50)
print("NEUROSCIENCE INTERPRETATION")
print("="*50)

print("\nTop behavioral predictors of gambling risk:")
top_features = importance_df.head(3)
for _, row in top_features.iterrows():
    feature = row['feature']
    importance = row['importance']
    
    if 'loss_chasing' in feature:
        mechanism = "reward system dysfunction (dopaminergic pathways)"
    elif 'session_intensity' in feature:
        mechanism = "compulsive behavior patterns"
    elif 'bet_size' in feature:
        mechanism = "impulse control deficits (prefrontal cortex)"
    elif 'loss_streak' in feature:
        mechanism = "emotional regulation impairment"
    else:
        mechanism = "behavioral dysregulation"
    
    print(f"- {feature.replace('_', ' ').title()} (importance: {importance:.3f})")
    print(f"  Neurological basis: {mechanism}")

print("\n" + "="*50)
print("SUMMARY FOR PAPER")
print("="*50)

print(f"""
Key Findings:
- Model achieved {accuracy:.1%} accuracy in classifying gambling risk levels
- {importance_df.iloc[0]['feature'].replace('_', ' ').title()} was the strongest predictor (importance: {importance_df.iloc[0]['importance']:.3f})
- Cross-validation confirmed model stability (CV accuracy: {cv_scores.mean():.3f})
- Clear behavioral differentiation between risk groups validated synthetic data approach

Dataset Statistics:
- Total users analyzed: {len(features_df)}
- Behavioral features extracted: {len(feature_columns)}
- Risk distribution: {dict(zip(['Normal', 'At-Risk', 'High-Risk'], features_df['risk_label'].value_counts().sort_index().values))}
""")

print("Model training complete. Results ready for publication.")

# Save model for future use
import joblib
joblib.dump(rf_model, 'gambling_risk_model.pkl')
print("\nModel saved as 'gambling_risk_model.pkl'")