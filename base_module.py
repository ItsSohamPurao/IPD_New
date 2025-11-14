"""
base_module.py

SHARED MODULE for all feature selection notebooks.
Place this file in the same directory as your notebooks.

DO NOT MODIFY THIS FILE after starting your experiments unless you re-run ALL notebooks.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
# CHANGED: Replaced OneHotEncoder with OrdinalEncoder
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# ========================================
# DATA LOADING
# ========================================

def load_and_clean_data(file_path):
    """
    Loads data, defines '?' as NaN, and drops identifiers and
    columns with >40% missing values.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, na_values='?')
    
    columns_to_drop = [
        'encounter_id', 'patient_nbr', 'weight', 'medical_specialty', 
        'payer_code', 'max_glu_serum', 'A1Cresult'
    ]
    
    key_features = ['race', 'diag_1', 'diag_2', 'diag_3', 'gender']
    df = df.drop(columns=columns_to_drop).dropna(subset=key_features)
    
    print(f"Cleaned data shape: {df.shape}")
    return df


# ========================================
# FEATURE ENGINEERING
# ========================================

def engineer_features(df):
    """
    Engineers features WITHOUT any encoding/scaling.
    """
    print("Engineering features...")
    
    df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    age_map = {
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
        '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
    }
    df['age_encoded'] = df['age'].map(age_map)

    def group_diagnosis(diag_code):
        diag_code = str(diag_code)
        if diag_code.startswith('250'): return 'Diabetes'
        elif 'V' in diag_code or 'E' in diag_code: return 'External'
        try:
            code = float(diag_code)
            if 390 <= code <= 459 or code == 785: return 'Circulatory'
            elif 460 <= code <= 519 or code == 786: return 'Respiratory'
            elif 520 <= code <= 579 or code == 787: return 'Digestive'
            elif 580 <= code <= 629 or code == 788: return 'Genitourinary'
            elif 140 <= code <= 239: return 'Neoplasms'
            elif 800 <= code <= 999: return 'Injury'
            elif 710 <= code <= 739: return 'Musculoskeletal'
            else: return 'Other'
        except ValueError:
            return 'Other'

    df['diag_1_grouped'] = df['diag_1'].apply(group_diagnosis)
    
    df = df.drop(columns=['readmitted', 'age', 'diag_1', 'diag_2', 'diag_3'])
    
    print(f"Engineered data shape: {df.shape}")
    return df


def get_feature_lists():
    """
    Returns lists of numeric and categorical features BEFORE encoding.
    """
    numeric_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses',
        'age_encoded'
    ]

    med_columns = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone', 
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]

    categorical_features = [
        'race', 'gender', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'change', 'diabetesMed',
        'diag_1_grouped'
    ] + med_columns
    
    return numeric_features, categorical_features


# ========================================
# PREPROCESSING & ENCODING
# ========================================

class DataPreprocessor:
    """
    Handles scaling and ordinal encoding.
    Maintains consistency across all notebooks.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        # CHANGED: Using OrdinalEncoder. 
        # It handles unknown values by assigning them -1.
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.numeric_features = None
        self.categorical_features = None
        self.all_encoded_feature_names = None
        self.feature_to_original_mapping = None
        
    def fit_transform(self, X_train, X_test, numeric_features, categorical_features):
        """
        Fit on training data and transform both train and test.
        
        Returns:
            X_train_encoded, X_test_encoded, all_encoded_feature_names, feature_mapping
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        print("\nPreprocessing data...")
        
        # Scale numeric
        X_train_numeric = self.scaler.fit_transform(X_train[numeric_features])
        X_test_numeric = self.scaler.transform(X_test[numeric_features])
        
        # CHANGED: Encode categorical features using OrdinalEncoder
        print(f"Ordinal Encoding {len(categorical_features)} features...")
        X_train_categorical = self.encoder.fit_transform(X_train[categorical_features])
        X_test_categorical = self.encoder.transform(X_test[categorical_features])
        
        # Get feature names
        numeric_names = numeric_features
        # CHANGED: With OrdinalEncoder, the feature names are just the original names
        categorical_names = categorical_features
        self.all_encoded_feature_names = numeric_names + categorical_names
        
        # Combine
        X_train_encoded = np.hstack([X_train_numeric, X_train_categorical])
        X_test_encoded = np.hstack([X_test_numeric, X_test_categorical])
        
        # CHANGED: Create simplified mapping
        self.feature_to_original_mapping = self._create_mapping(
            numeric_names, categorical_names
        )
        
        print(f"Encoded features: {X_train_encoded.shape[1]}")
        print(f"  - Numeric: {len(numeric_names)}")
        print(f"  - Categorical (Ordinal): {len(categorical_names)}") # Updated text
        
        return X_train_encoded, X_test_encoded, self.all_encoded_feature_names, self.feature_to_original_mapping
    
    def _create_mapping(self, numeric_names, categorical_names):
        """Creates mapping from encoded features to original features."""
        mapping = {}
        
        # Numeric features map to themselves
        for feat in numeric_names:
            mapping[feat] = feat
        
        # CHANGED: With OrdinalEncoder, categorical features also map to themselves,
        # as no new feature columns are created.
        for feat in categorical_names:
            mapping[feat] = feat
        
        return mapping


# ========================================
# RESULTS SAVING
# ========================================

def save_model_results(algorithm_name, selected_encoded_features, all_encoded_features, 
                       all_original_features, feature_mapping, report, 
                       results_csv='study_results.csv'):
    """
    Saves results with BOTH encoded and original feature tracking.
    
    Args:
        algorithm_name: Name of the FS algorithm
        selected_encoded_features: List of selected encoded feature names
        all_encoded_features: List of all encoded feature names
        all_original_features: List of all original engineered features
        feature_mapping: Dict mapping encoded -> original
        report: sklearn classification_report dict
        results_csv: Path to master results CSV
    """
    print(f"\nSaving results for: {algorithm_name}")
    
    # Map to original features
    selected_original = set()
    for encoded_feat in selected_encoded_features:
        original = feature_mapping.get(encoded_feat, encoded_feat)
        selected_original.add(original)
    
    selected_original = sorted(list(selected_original))
    
    # Counts
    selected_encoded_count = len(selected_encoded_features)
    selected_original_count = len(selected_original)
    total_encoded = len(all_encoded_features)
    total_original = len(all_original_features)
    
    # CSV data
    report_flat = {
        'algorithm': algorithm_name,
        'features_selected_count': selected_original_count,
        'features_discarded_count': total_original - selected_original_count,
        'total_features_engineered': total_original,
        'encoded_features_selected': selected_encoded_count,
        'encoded_features_total': total_encoded,
        'accuracy': report['accuracy'],
        'precision_0': report['0']['precision'],
        'recall_0': report['0']['recall'],
        'f1_score_0': report['0']['f1-score'],
        'precision_1': report['1']['precision'],
        'recall_1': report['1']['recall'],
        'f1_score_1': report['1']['f1-score'],
    }
    
    results_df = pd.DataFrame([report_flat])
    if not os.path.exists(results_csv):
        results_df.to_csv(results_csv, index=False)
    else:
        results_df.to_csv(results_csv, mode='a', header=False, index=False)
    
    print(f"✓ Appended to {results_csv}")

    # JSON data
    discarded_original = sorted(list(set(all_original_features) - set(selected_original)))
    discarded_encoded = sorted(list(set(all_encoded_features) - set(selected_encoded_features)))
    
    json_data = {
        'algorithm_name': algorithm_name,
        'model_scores': report,
        'original_features': {
            'selected_count': selected_original_count,
            'discarded_count': len(discarded_original),
            'total_count': total_original,
            'selected': selected_original,
            'discarded': discarded_original
        },
        'encoded_features': {
            'selected_count': selected_encoded_count,
            'discarded_count': len(discarded_encoded),
            'total_count': total_encoded,
            'selected': sorted(selected_encoded_features),
            'discarded': discarded_encoded
        }
    }
    
    json_filename = f"{algorithm_name.lower().replace(' ', '_')}_results.json"
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print(f"✓ Saved to {json_filename}")
    print(f"\n  Original: {selected_original_count}/{total_original} features")
    print(f"  Encoded:  {selected_encoded_count}/{total_encoded} features")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  F1 (class 1): {report['1']['f1-score']:.4f}")


# ========================================
# STANDARD TRAINING FUNCTION
# ========================================

def train_and_evaluate(X_train, X_test, y_train, y_test, 
                       kernel='linear', C=1, max_iter=10000, random_state=42):
    """
    Standard SVM training and evaluation.
    Use this in all notebooks for consistency.
    
    Returns:
        clf, y_pred, report
    """
    print("\nTraining SVM classifier...")
    clf = SVC(kernel=kernel, C=C, max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['0', '1'], output_dict=True)
    
    return clf, y_pred, report


# ========================================
# COMPLETE PIPELINE
# ========================================

def get_preprocessed_data(data_path, test_size=0.2, random_state=42):
    """
    Complete pipeline from raw data to encoded train/test sets.
    Use this at the start of each notebook.
    
    Returns:
        X_train_encoded, X_test_encoded, y_train, y_test,
        all_encoded_features, all_original_features, feature_mapping
    """
    # Load and engineer
    df = load_and_clean_data(data_path)
    df_engineered = engineer_features(df.copy())
    
    # Get feature lists
    numeric_features, categorical_features = get_feature_lists()
    all_original_features = numeric_features + categorical_features
    
    # Prepare data
    for col in categorical_features:
        df_engineered[col] = df_engineered[col].astype(str)
    
    X = df_engineered[all_original_features]
    y = df_engineered['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train_encoded, X_test_encoded, all_encoded_features, feature_mapping = \
        preprocessor.fit_transform(X_train, X_test, numeric_features, categorical_features)
    
    return (X_train_encoded, X_test_encoded, y_train, y_test,
            all_encoded_features, all_original_features, feature_mapping)


# ========================================
# CONSTANTS
# ========================================

# Use these in all notebooks for consistency
RANDOM_STATE = 42
TEST_SIZE = 0.2
SVM_PARAMS = {
    'kernel': 'linear',
    'C': 1,
    'max_iter': 10000,
    'random_state': RANDOM_STATE
}