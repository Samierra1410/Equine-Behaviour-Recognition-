import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, precision_recall_fscore_support
)
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("🐴 TRUE HIERARCHICAL CATBOOST - 6 SUB-BEHAVIORS 🐴")
print("="*80)
print("\nStage 1: Parent classification (affiliative/neutral/avoidant)")
print("Stage 2: Sub-behavior classification within each parent")
print("="*80)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ============================================================================
# LOAD & PREPARE DATA
# ============================================================================

df = pd.read_csv(BASE_DIR / "data/ultimate_combined" / "all_features_ULTIMATE.csv")

if 'grouped_behavior' not in df.columns:
    def map_to_grouped(row):
        behavior = str(row.get('behavior', '')).lower()
        category = str(row.get('category', '')).lower()
        is_horse = 'horse' in behavior
        is_active = any(w in behavior for w in ['active', 'engagement', 'nose', 'joint', 'touch'])
        
        if category == 'affiliative':
            return 'affiliative-active' if is_active else 'affiliative-subtle'
        elif category == 'neutral':
            return 'neutral-horse' if is_horse else 'neutral-human'
        elif category == 'avoidant':
            return 'avoidant-horse' if is_horse else 'avoidant-human'
        return 'unknown'
    
    df['grouped_behavior'] = df.apply(map_to_grouped, axis=1)

# Add parent category
def get_parent(behavior):
    return behavior.split('-')[0]

df['parent_category'] = df['grouped_behavior'].apply(get_parent)

valid_behaviors = [
    'affiliative-active', 'affiliative-subtle',
    'neutral-horse', 'neutral-human',
    'avoidant-horse', 'avoidant-human'
]

df = df[df['grouped_behavior'].isin(valid_behaviors)].reset_index(drop=True)

print(f"\n✅ Loaded: {len(df):,} samples")
print(f"\nDistribution:")
for parent in ['affiliative', 'neutral', 'avoidant']:
    count = (df['parent_category'] == parent).sum()
    pct = count / len(df) * 100
    print(f"  {parent:12}: {count:6,} ({pct:5.1f}%)")

# ============================================================================
# FEATURE PREPARATION
# ============================================================================

exclude = ['video', 'timestamp', 'frame', 'behavior', 'category', 
           'grouped_behavior', 'parent_category', 'event_type', 'boris_event_timestamp']

feature_cols = [c for c in df.columns 
                if c not in exclude 
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

feature_cols = [c for c in feature_cols if df[c].nunique() > 1]

X = df[feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n✅ Features: {len(feature_cols)}")

# ============================================================================
# STAGE 1: PARENT CATEGORY CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("STAGE 1: PARENT CATEGORY CLASSIFICATION")
print("="*80)

# Encode parent categories
le_parent = LabelEncoder()
y_parent = le_parent.fit_transform(df['parent_category'])

print(f"\nTraining Stage 1 model (3 parent categories)...")

# Train Stage 1 model
stage1_model = CatBoostClassifier(
    iterations=400,
    depth=10,
    learning_rate=0.03,
    auto_class_weights='Balanced',
    verbose=False,
    random_state=SEED
)

# Cross-validation predictions for Stage 1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
stage1_predictions = cross_val_predict(stage1_model, X_scaled, y_parent, cv=skf, n_jobs=-1)

# Stage 1 metrics
stage1_bal_acc = balanced_accuracy_score(y_parent, stage1_predictions)
stage1_acc = accuracy_score(y_parent, stage1_predictions)
stage1_f1 = f1_score(y_parent, stage1_predictions, average='weighted')

print(f"\n✅ Stage 1 Results:")
print(f"   Balanced Accuracy: {stage1_bal_acc*100:.2f}%")
print(f"   Regular Accuracy:  {stage1_acc*100:.2f}%")
print(f"   Weighted F1:       {stage1_f1*100:.2f}%")

print(f"\n   Classification Report:")
print(classification_report(y_parent, stage1_predictions, 
                           target_names=le_parent.classes_, zero_division=0))

# ============================================================================
# STAGE 2: SUB-BEHAVIOR CLASSIFICATION (PER PARENT)
# ============================================================================

print("\n" + "="*80)
print("STAGE 2: SUB-BEHAVIOR CLASSIFICATION")
print("="*80)

# Store Stage 2 models and predictions
stage2_models = {}
stage2_predictions_dict = {}
stage2_metrics = {}

# Final hierarchical predictions (will be filled)
final_predictions = np.zeros(len(df), dtype=int)

# Label encoder for full 6 behaviors
le_full = LabelEncoder()
le_full.fit(df['grouped_behavior'])

# ============================================================================
# STAGE 2A: AFFILIATIVE SUB-BEHAVIORS
# ============================================================================

print("\n" + "-"*80)
print("Stage 2A: Affiliative Sub-behaviors (active vs subtle)")
print("-"*80)

aff_mask = df['parent_category'] == 'affiliative'
X_aff = X_scaled[aff_mask]
y_aff_labels = df.loc[aff_mask, 'grouped_behavior']

le_aff = LabelEncoder()
y_aff = le_aff.fit_transform(y_aff_labels)

print(f"\n   Samples: {len(y_aff):,}")
print(f"   Classes: {le_aff.classes_}")

if len(np.unique(y_aff)) > 1:
    stage2_aff_model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.03,
        auto_class_weights='Balanced',
        verbose=False,
        random_state=SEED
    )
    
    skf_aff = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    stage2_aff_preds = cross_val_predict(stage2_aff_model, X_aff, y_aff, cv=skf_aff, n_jobs=-1)
    
    aff_bal_acc = balanced_accuracy_score(y_aff, stage2_aff_preds)
    aff_acc = accuracy_score(y_aff, stage2_aff_preds)
    aff_f1 = f1_score(y_aff, stage2_aff_preds, average='weighted')
    
    print(f"\n   ✅ Balanced Accuracy: {aff_bal_acc*100:.2f}%")
    print(f"      Regular Accuracy:  {aff_acc*100:.2f}%")
    print(f"      Weighted F1:       {aff_f1*100:.2f}%")
    
    stage2_models['affiliative'] = stage2_aff_model
    stage2_predictions_dict['affiliative'] = stage2_aff_preds
    stage2_metrics['affiliative'] = {
        'balanced_accuracy': aff_bal_acc,
        'accuracy': aff_acc,
        'f1': aff_f1
    }
    
    # Map back to full label space
    aff_indices = np.where(aff_mask)[0]
    for idx, pred in zip(aff_indices, stage2_aff_preds):
        pred_label = le_aff.inverse_transform([pred])[0]
        final_predictions[idx] = le_full.transform([pred_label])[0]

# ============================================================================
# STAGE 2B: NEUTRAL SUB-BEHAVIORS
# ============================================================================

print("\n" + "-"*80)
print("Stage 2B: Neutral Sub-behaviors (horse vs human)")
print("-"*80)

neu_mask = df['parent_category'] == 'neutral'
X_neu = X_scaled[neu_mask]
y_neu_labels = df.loc[neu_mask, 'grouped_behavior']

le_neu = LabelEncoder()
y_neu = le_neu.fit_transform(y_neu_labels)

print(f"\n   Samples: {len(y_neu):,}")
print(f"   Classes: {le_neu.classes_}")

if len(np.unique(y_neu)) > 1:
    stage2_neu_model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.03,
        auto_class_weights='Balanced',
        verbose=False,
        random_state=SEED
    )
    
    skf_neu = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    stage2_neu_preds = cross_val_predict(stage2_neu_model, X_neu, y_neu, cv=skf_neu, n_jobs=-1)
    
    neu_bal_acc = balanced_accuracy_score(y_neu, stage2_neu_preds)
    neu_acc = accuracy_score(y_neu, stage2_neu_preds)
    neu_f1 = f1_score(y_neu, stage2_neu_preds, average='weighted')
    
    print(f"\n   ✅ Balanced Accuracy: {neu_bal_acc*100:.2f}%")
    print(f"      Regular Accuracy:  {neu_acc*100:.2f}%")
    print(f"      Weighted F1:       {neu_f1*100:.2f}%")
    
    stage2_models['neutral'] = stage2_neu_model
    stage2_predictions_dict['neutral'] = stage2_neu_preds
    stage2_metrics['neutral'] = {
        'balanced_accuracy': neu_bal_acc,
        'accuracy': neu_acc,
        'f1': neu_f1
    }
    
    # Map back to full label space
    neu_indices = np.where(neu_mask)[0]
    for idx, pred in zip(neu_indices, stage2_neu_preds):
        pred_label = le_neu.inverse_transform([pred])[0]
        final_predictions[idx] = le_full.transform([pred_label])[0]

# ============================================================================
# STAGE 2C: AVOIDANT SUB-BEHAVIORS
# ============================================================================

print("\n" + "-"*80)
print("Stage 2C: Avoidant Sub-behaviors (horse vs human)")
print("-"*80)

avo_mask = df['parent_category'] == 'avoidant'
X_avo = X_scaled[avo_mask]
y_avo_labels = df.loc[avo_mask, 'grouped_behavior']

le_avo = LabelEncoder()
y_avo = le_avo.fit_transform(y_avo_labels)

print(f"\n   Samples: {len(y_avo):,}")
print(f"   Classes: {le_avo.classes_}")

if len(np.unique(y_avo)) > 1:
    stage2_avo_model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.03,
        auto_class_weights='Balanced',
        verbose=False,
        random_state=SEED
    )
    
    skf_avo = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    stage2_avo_preds = cross_val_predict(stage2_avo_model, X_avo, y_avo, cv=skf_avo, n_jobs=-1)
    
    avo_bal_acc = balanced_accuracy_score(y_avo, stage2_avo_preds)
    avo_acc = accuracy_score(y_avo, stage2_avo_preds)
    avo_f1 = f1_score(y_avo, stage2_avo_preds, average='weighted')
    
    print(f"\n   ✅ Balanced Accuracy: {avo_bal_acc*100:.2f}%")
    print(f"      Regular Accuracy:  {avo_acc*100:.2f}%")
    print(f"      Weighted F1:       {avo_f1*100:.2f}%")
    
    stage2_models['avoidant'] = stage2_avo_model
    stage2_predictions_dict['avoidant'] = stage2_avo_preds
    stage2_metrics['avoidant'] = {
        'balanced_accuracy': avo_bal_acc,
        'accuracy': avo_acc,
        'f1': avo_f1
    }
    
    # Map back to full label space
    avo_indices = np.where(avo_mask)[0]
    for idx, pred in zip(avo_indices, stage2_avo_preds):
        pred_label = le_avo.inverse_transform([pred])[0]
        final_predictions[idx] = le_full.transform([pred_label])[0]

# ============================================================================
# FINAL HIERARCHICAL RESULTS
# ============================================================================

print("\n\n" + "="*80)
print("🎉 FINAL HIERARCHICAL RESULTS (6 SUB-BEHAVIORS)")
print("="*80)

# True labels
y_true = le_full.transform(df['grouped_behavior'])

# Overall metrics
final_bal_acc = balanced_accuracy_score(y_true, final_predictions)
final_acc = accuracy_score(y_true, final_predictions)
final_f1 = f1_score(y_true, final_predictions, average='weighted')

print(f"\n🏆 OVERALL HIERARCHICAL METRICS:")
print(f"   Balanced Accuracy: {final_bal_acc*100:.2f}%")
print(f"   Regular Accuracy:  {final_acc*100:.2f}%")
print(f"   Weighted F1 Score: {final_f1*100:.2f}%")

print(f"\n📋 Classification Report:")
print(classification_report(y_true, final_predictions, 
                           target_names=le_full.classes_, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_true, final_predictions)
precision, recall, f1_per, support = precision_recall_fscore_support(y_true, final_predictions, zero_division=0)

# ============================================================================
# DETAILED ACCURACY & F1 BREAKDOWN
# ============================================================================

print(f"\n{'='*80}")
print("📊 DETAILED HIERARCHICAL ACCURACY & F1 METRICS")
print("="*80)

# 1. OVERALL
print(f"\n1️⃣  OVERALL (All 6 Behaviors - Hierarchical):")
print(f"    Accuracy: {final_acc*100:.2f}%")
print(f"    F1 Score: {final_f1*100:.2f}%")

# 2. AFFILIATIVE (combined)
aff_mask_full = df['parent_category'] == 'affiliative'
if aff_mask_full.sum() > 0:
    aff_y_full = y_true[aff_mask_full]
    aff_pred_full = final_predictions[aff_mask_full]
    aff_acc_combined = accuracy_score(aff_y_full, aff_pred_full)
    aff_f1_combined = f1_score(aff_y_full, aff_pred_full, average='weighted')
    print(f"\n2️⃣  AFFILIATIVE (Combined - both sub-behaviors):")
    print(f"    Accuracy: {aff_acc_combined*100:.2f}%")
    print(f"    F1 Score: {aff_f1_combined*100:.2f}%")
    print(f"    Samples:  {aff_mask_full.sum():,}")
    print(f"    Stage 2 Accuracy: {stage2_metrics['affiliative']['accuracy']*100:.2f}%")

# 3. NEUTRAL (combined)
neu_mask_full = df['parent_category'] == 'neutral'
if neu_mask_full.sum() > 0:
    neu_y_full = y_true[neu_mask_full]
    neu_pred_full = final_predictions[neu_mask_full]
    neu_acc_combined = accuracy_score(neu_y_full, neu_pred_full)
    neu_f1_combined = f1_score(neu_y_full, neu_pred_full, average='weighted')
    print(f"\n3️⃣  NEUTRAL (Combined - both sub-behaviors):")
    print(f"    Accuracy: {neu_acc_combined*100:.2f}%")
    print(f"    F1 Score: {neu_f1_combined*100:.2f}%")
    print(f"    Samples:  {neu_mask_full.sum():,}")
    print(f"    Stage 2 Accuracy: {stage2_metrics['neutral']['accuracy']*100:.2f}%")

# 4. AVOIDANT (combined)
avo_mask_full = df['parent_category'] == 'avoidant'
if avo_mask_full.sum() > 0:
    avo_y_full = y_true[avo_mask_full]
    avo_pred_full = final_predictions[avo_mask_full]
    avo_acc_combined = accuracy_score(avo_y_full, avo_pred_full)
    avo_f1_combined = f1_score(avo_y_full, avo_pred_full, average='weighted')
    print(f"\n4️⃣  AVOIDANT (Combined - both sub-behaviors):")
    print(f"    Accuracy: {avo_acc_combined*100:.2f}%")
    print(f"    F1 Score: {avo_f1_combined*100:.2f}%")
    print(f"    Samples:  {avo_mask_full.sum():,}")
    print(f"    Stage 2 Accuracy: {stage2_metrics['avoidant']['accuracy']*100:.2f}%")

# 5-7. Individual sub-behaviors
print(f"\n5️⃣  AFFILIATIVE SUB-BEHAVIORS (Individual):")
for behavior in ['affiliative-active', 'affiliative-subtle']:
    idx = list(le_full.classes_).index(behavior)
    mask = y_true == idx
    if mask.sum() > 0:
        correct = (final_predictions[mask] == idx).sum()
        acc_indiv = correct / mask.sum()
        f1_indiv = f1_per[idx]
        print(f"\n    {'a)' if 'active' in behavior else 'b)'} {behavior}:")
        print(f"       Accuracy: {acc_indiv*100:.2f}%")
        print(f"       F1 Score: {f1_indiv*100:.2f}%")
        print(f"       Samples:  {mask.sum():,}")

print(f"\n6️⃣  NEUTRAL SUB-BEHAVIORS (Individual):")
for behavior in ['neutral-horse', 'neutral-human']:
    idx = list(le_full.classes_).index(behavior)
    mask = y_true == idx
    if mask.sum() > 0:
        correct = (final_predictions[mask] == idx).sum()
        acc_indiv = correct / mask.sum()
        f1_indiv = f1_per[idx]
        print(f"\n    {'a)' if 'horse' in behavior else 'b)'} {behavior}:")
        print(f"       Accuracy: {acc_indiv*100:.2f}%")
        print(f"       F1 Score: {f1_indiv*100:.2f}%")
        print(f"       Samples:  {mask.sum():,}")

print(f"\n7️⃣  AVOIDANT SUB-BEHAVIORS (Individual):")
for behavior in ['avoidant-horse', 'avoidant-human']:
    idx = list(le_full.classes_).index(behavior)
    mask = y_true == idx
    if mask.sum() > 0:
        correct = (final_predictions[mask] == idx).sum()
        acc_indiv = correct / mask.sum()
        f1_indiv = f1_per[idx]
        print(f"\n    {'a)' if 'horse' in behavior else 'b)'} {behavior}:")
        print(f"       Accuracy: {acc_indiv*100:.2f}%")
        print(f"       F1 Score: {f1_indiv*100:.2f}%")
        print(f"       Samples:  {mask.sum():,}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print(f"\n{'='*80}")
print("📊 HIERARCHICAL SUMMARY TABLE")
print("="*80)

# Calculate individual accuracies
acc_aff_act = (final_predictions[y_true == le_full.transform(['affiliative-active'])[0]] == le_full.transform(['affiliative-active'])[0]).sum() / (y_true == le_full.transform(['affiliative-active'])[0]).sum()
acc_aff_sub = (final_predictions[y_true == le_full.transform(['affiliative-subtle'])[0]] == le_full.transform(['affiliative-subtle'])[0]).sum() / (y_true == le_full.transform(['affiliative-subtle'])[0]).sum()
acc_neu_hor = (final_predictions[y_true == le_full.transform(['neutral-horse'])[0]] == le_full.transform(['neutral-horse'])[0]).sum() / (y_true == le_full.transform(['neutral-horse'])[0]).sum()
acc_neu_hum = (final_predictions[y_true == le_full.transform(['neutral-human'])[0]] == le_full.transform(['neutral-human'])[0]).sum() / (y_true == le_full.transform(['neutral-human'])[0]).sum()
acc_avo_hor = (final_predictions[y_true == le_full.transform(['avoidant-horse'])[0]] == le_full.transform(['avoidant-horse'])[0]).sum() / (y_true == le_full.transform(['avoidant-horse'])[0]).sum()
acc_avo_hum = (final_predictions[y_true == le_full.transform(['avoidant-human'])[0]] == le_full.transform(['avoidant-human'])[0]).sum() / (y_true == le_full.transform(['avoidant-human'])[0]).sum()

f1_aff_act = f1_per[le_full.transform(['affiliative-active'])[0]]
f1_aff_sub = f1_per[le_full.transform(['affiliative-subtle'])[0]]
f1_neu_hor = f1_per[le_full.transform(['neutral-horse'])[0]]
f1_neu_hum = f1_per[le_full.transform(['neutral-human'])[0]]
f1_avo_hor = f1_per[le_full.transform(['avoidant-horse'])[0]]
f1_avo_hum = f1_per[le_full.transform(['avoidant-human'])[0]]

summary_data = [
    ["OVERALL (Hierarchical)", final_acc*100, final_f1*100, len(y_true), ""],
    ["", "", "", "", ""],
    ["STAGE 1 (Parent Categories)", "", "", "", ""],
    ["  └─ 3-way classification", stage1_acc*100, stage1_f1*100, len(y_parent), f"{stage1_bal_acc*100:.2f}%"],
    ["", "", "", "", ""],
    ["AFFILIATIVE (combined)", aff_acc_combined*100, aff_f1_combined*100, aff_mask_full.sum(), ""],
    ["  ├─ Stage 2 accuracy", stage2_metrics['affiliative']['accuracy']*100, stage2_metrics['affiliative']['f1']*100, "", ""],
    ["  ├─ affiliative-active", acc_aff_act*100, f1_aff_act*100, (y_true == le_full.transform(['affiliative-active'])[0]).sum(), ""],
    ["  └─ affiliative-subtle", acc_aff_sub*100, f1_aff_sub*100, (y_true == le_full.transform(['affiliative-subtle'])[0]).sum(), ""],
    ["", "", "", "", ""],
    ["NEUTRAL (combined)", neu_acc_combined*100, neu_f1_combined*100, neu_mask_full.sum(), ""],
    ["  ├─ Stage 2 accuracy", stage2_metrics['neutral']['accuracy']*100, stage2_metrics['neutral']['f1']*100, "", ""],
    ["  ├─ neutral-horse", acc_neu_hor*100, f1_neu_hor*100, (y_true == le_full.transform(['neutral-horse'])[0]).sum(), ""],
    ["  └─ neutral-human", acc_neu_hum*100, f1_neu_hum*100, (y_true == le_full.transform(['neutral-human'])[0]).sum(), ""],
    ["", "", "", "", ""],
    ["AVOIDANT (combined)", avo_acc_combined*100, avo_f1_combined*100, avo_mask_full.sum(), ""],
    ["  ├─ Stage 2 accuracy", stage2_metrics['avoidant']['accuracy']*100, stage2_metrics['avoidant']['f1']*100, "", ""],
    ["  ├─ avoidant-horse", acc_avo_hor*100, f1_avo_hor*100, (y_true == le_full.transform(['avoidant-horse'])[0]).sum(), ""],
    ["  └─ avoidant-human", acc_avo_hum*100, f1_avo_hum*100, (y_true == le_full.transform(['avoidant-human'])[0]).sum(), ""],
]

print(f"{'Category':<35} {'Accuracy':>12} {'F1 Score':>12} {'Samples':>10} {'Bal.Acc':>10}")
print("-" * 82)
for row in summary_data:
    if row[0] == "":
        print()
    else:
        cat, acc_val, f1_val, samp, bal = row
        if isinstance(acc_val, str):
            print(f"{cat:<35} {acc_val:>12} {f1_val:>12} {samp:>10} {bal:>10}")
        else:
            samp_str = f"{samp:,}" if samp != "" else ""
            print(f"{cat:<35} {acc_val:>11.2f}% {f1_val:>11.2f}% {samp_str:>10} {bal:>10}")

print("="*80)


# ============================================================================
# CASCADED END-TO-END EVALUATION (Reviewer 1 Request)
# ============================================================================
print("\n" + "="*80)
print("CASCADED END-TO-END EVALUATION")
print("Stage 1 PREDICTIONS route to Stage 2 (not ground truth)")
print("="*80)

cascaded_predictions = np.zeros(len(df), dtype=int)

# Use Stage 1 PREDICTIONS to route
predicted_parents = le_parent.inverse_transform(stage1_predictions)

for parent in ['affiliative', 'neutral', 'avoidant']:
    # Samples that Stage 1 PREDICTED as this parent
    routed_mask = predicted_parents == parent
    if routed_mask.sum() == 0:
        continue
    
    X_routed = X_scaled[routed_mask]
    
    # Use the Stage 2 model for this parent
    if parent in stage2_models:
        le_sub = {'affiliative': le_aff, 'neutral': le_neu, 'avoidant': le_avo}[parent]
        model = stage2_models[parent]
        model.fit(X_scaled[df['parent_category'] == parent], 
                  {'affiliative': y_aff, 'neutral': y_neu, 'avoidant': y_avo}[parent])
        sub_preds = model.predict(X_routed).flatten().astype(int)
        
        routed_indices = np.where(routed_mask)[0]
        for idx, pred in zip(routed_indices, sub_preds):
            pred_label = le_sub.inverse_transform([pred])[0]
            cascaded_predictions[idx] = le_full.transform([pred_label])[0]

# Cascaded metrics
casc_bal_acc = balanced_accuracy_score(y_true, cascaded_predictions)
casc_acc = accuracy_score(y_true, cascaded_predictions)
casc_f1 = f1_score(y_true, cascaded_predictions, average='weighted')

print(f"\nCASCADED vs ORACLE:")
print(f"  Oracle   Bal.Acc: {final_bal_acc*100:.1f}%  Acc: {final_acc*100:.1f}%")
print(f"  Cascaded Bal.Acc: {casc_bal_acc*100:.1f}%  Acc: {casc_acc*100:.1f}%")
print(f"  Drop:            {(final_bal_acc-casc_bal_acc)*100:.1f} pp")

# Cascaded confusion matrix
cm_cascaded = confusion_matrix(y_true, cascaded_predictions)
print(f"\nCascaded Classification Report:")
print(classification_report(y_true, cascaded_predictions,
                           target_names=le_full.classes_, zero_division=0))

# Cross-parent errors
cross_parent_errors = 0
for i, (true, pred) in enumerate(zip(y_true, cascaded_predictions)):
    true_parent = le_full.inverse_transform([true])[0].split('-')[0]
    pred_parent = le_full.inverse_transform([pred])[0].split('-')[0]
    if true_parent != pred_parent:
        cross_parent_errors += 1

print(f"Cross-parent misclassifications: {cross_parent_errors} ({cross_parent_errors/len(y_true)*100:.1f}%)")

# Save cascaded confusion matrix figure
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=le_full.classes_, yticklabels=le_full.classes_)
axes[0].set_title(f'Oracle Routing\nBal.Acc: {final_bal_acc*100:.1f}%')
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')

sns.heatmap(cm_cascaded, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=le_full.classes_, yticklabels=le_full.classes_)
axes[1].set_title(f'Cascaded (End-to-End)\nBal.Acc: {casc_bal_acc*100:.1f}%')
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.suptitle('Oracle vs Cascaded Confusion Matrices', fontsize=14, fontweight='bold')
plt.savefig(OUTPUT_DIR / 'CASCADED_vs_ORACLE.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(24, 8))
gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

# 1. Stage 1 confusion matrix
ax1 = fig.add_subplot(gs[0, 0])
cm_stage1 = confusion_matrix(y_parent, stage1_predictions)
sns.heatmap(cm_stage1, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=le_parent.classes_, yticklabels=le_parent.classes_)
ax1.set_title(f'Stage 1: Parent Categories\nAcc: {stage1_bal_acc*100:.1f}%')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')

# 2. Final hierarchical confusion matrix
ax2 = fig.add_subplot(gs[:, 1:3])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=le_full.classes_, yticklabels=le_full.classes_)
ax2.set_title(f'Final Hierarchical: 6 Sub-Behaviors\nBalanced Acc: {final_bal_acc*100:.1f}%')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax2.get_yticklabels(), rotation=0)

# 3. Stage 2 accuracies
ax3 = fig.add_subplot(gs[0, 3])
stage2_names = ['Affiliative', 'Neutral', 'Avoidant']
stage2_accs = [
    stage2_metrics['affiliative']['accuracy'] * 100,
    stage2_metrics['neutral']['accuracy'] * 100,
    stage2_metrics['avoidant']['accuracy'] * 100
]
colors = ['#ff6b6b', '#4ecdc4', '#ffe66d']
ax3.bar(stage2_names, stage2_accs, color=colors)
ax3.set_ylabel('Accuracy (%)')
ax3.set_title('Stage 2: Sub-behavior Accuracy')
ax3.set_ylim([0, 100])
for i, v in enumerate(stage2_accs):
    ax3.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# 4. Per-class recall
ax4 = fig.add_subplot(gs[1, 0])
recalls = [recall[i] * 100 for i in range(len(le_full.classes_))]
ax4.barh(le_full.classes_, recalls, color='green')
ax4.set_xlabel('Recall (%)')
ax4.set_title('Per-Class Recall')
ax4.set_xlim([0, 100])

# 5. Hierarchical flow diagram (text)
ax5 = fig.add_subplot(gs[1, 3])
ax5.axis('off')
flow_text = f"""HIERARCHICAL FLOW:

Stage 1 (3 classes):
  Acc: {stage1_acc*100:.1f}%
  
Stage 2A (Affiliative):
  Acc: {stage2_metrics['affiliative']['accuracy']*100:.1f}%
  
Stage 2B (Neutral):
  Acc: {stage2_metrics['neutral']['accuracy']*100:.1f}%
  
Stage 2C (Avoidant):
  Acc: {stage2_metrics['avoidant']['accuracy']*100:.1f}%

FINAL:
  Acc: {final_acc*100:.1f}%
  Bal.Acc: {final_bal_acc*100:.1f}%
"""
ax5.text(0.1, 0.9, flow_text, transform=ax5.transAxes, 
         fontfamily='monospace', fontsize=10, verticalalignment='top')

plt.suptitle('TRUE HIERARCHICAL CLASSIFICATION RESULTS', fontsize=16, fontweight='bold')
plt.savefig(OUTPUT_DIR / 'HIERARCHICAL_6BEH_RESULTS.png', dpi=200, bbox_inches='tight')
plt.close()

print(f"\n✅ Visualization saved!")

# ============================================================================
# SAVE MODELS
# ============================================================================

print(f"\n💾 Saving models...")

# Retrain final models on full data
stage1_final = CatBoostClassifier(iterations=400, depth=10, learning_rate=0.03,
                                   auto_class_weights='Balanced', verbose=False, random_state=SEED)
stage1_final.fit(X_scaled, y_parent)
stage1_final.save_model(str(OUTPUT_DIR / 'stage1_parent_model.cbm'))

# Retrain Stage 2 models
for parent in ['affiliative', 'neutral', 'avoidant']:
    mask = df['parent_category'] == parent
    X_sub = X_scaled[mask]
    y_sub_labels = df.loc[mask, 'grouped_behavior']
    le_sub = LabelEncoder()
    y_sub = le_sub.fit_transform(y_sub_labels)
    
    if len(np.unique(y_sub)) > 1:
        stage2_final = CatBoostClassifier(iterations=300, depth=8, learning_rate=0.03,
                                          auto_class_weights='Balanced', verbose=False, random_state=SEED)
        stage2_final.fit(X_sub, y_sub)
        stage2_final.save_model(str(OUTPUT_DIR / f'stage2_{parent}_model.cbm'))

# Save metadata
metadata = {
    'stage1': {
        'classes': le_parent.classes_.tolist(),
        'balanced_accuracy': float(stage1_bal_acc),
        'accuracy': float(stage1_acc),
        'f1': float(stage1_f1)
    },
    'stage2': stage2_metrics,
    'final': {
        'classes': le_full.classes_.tolist(),
        'balanced_accuracy': float(final_bal_acc),
        'accuracy': float(final_acc),
        'f1': float(final_f1)
    },
    'feature_names': feature_cols
}

with open(OUTPUT_DIR / 'hierarchical_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*80}")
print(f"✅ COMPLETE! All results saved to: {OUTPUT_DIR}")
print(f"\nSaved files:")
print(f"  • stage1_parent_model.cbm (3 parent categories)")
print(f"  • stage2_affiliative_model.cbm")
print(f"  • stage2_neutral_model.cbm")
print(f"  • stage2_avoidant_model.cbm")
print(f"  • hierarchical_metadata.json")
print(f"  • HIERARCHICAL_6BEH_RESULTS.png")
print("="*80)