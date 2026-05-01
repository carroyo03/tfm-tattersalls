import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression, Ridge

# ── Configuration ──
DATA_DIR = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42

# ── Best Params (extracted from MLflow) ──
CLF_PARAMS = {
    "rf": {'n_estimators': 461, 'max_depth': 6, 'min_samples_split': 14, 'min_samples_leaf': 3, 'max_features': 0.5},
    "xgb": {'n_estimators': 580, 'learning_rate': 0.1082, 'max_depth': 4, 'reg_lambda': 0.8574, 'subsample': 0.6685, 'colsample_bytree': 0.6335},
    "lgbm": {'n_estimators': 711, 'learning_rate': 0.0112, 'num_leaves': 300, 'reg_lambda': 0.2625, 'subsample': 0.6035, 'colsample_bytree': 0.9768, 'min_child_samples': 74},
    "cb": {'iterations': 486, 'learning_rate': 0.0102, 'depth': 10, 'l2_leaf_reg': 1.0422, 'bagging_temperature': 0.8336}
}

REG_PARAMS = {
    "rf": {'n_estimators': 486, 'max_depth': 11, 'min_samples_split': 2, 'min_samples_leaf': 8, 'max_features': 0.5},
    "xgb": {'n_estimators': 637, 'learning_rate': 0.1397, 'max_depth': 4, 'reg_lambda': 0.2208, 'subsample': 0.6041, 'colsample_bytree': 0.6723},
    "lgbm": {'n_estimators': 253, 'learning_rate': 0.1120, 'num_leaves': 71, 'reg_lambda': 0.1501, 'subsample': 0.7283, 'colsample_bytree': 0.5891, 'min_child_samples': 94},
    "cb": {'iterations': 833, 'learning_rate': 0.0742, 'depth': 5, 'l2_leaf_reg': 4.4945}
}

# ── Stacking Weights (from MLflow) ──
# We will use these to reconstruct the meta-learner if we don't want to re-fit
# Stage 1: LogisticRegression
# Stage 2: Ridge
CLF_WEIGHTS = {"rf": 1.1616, "xgb": 0.5265, "lgbm": 0.6199, "cb": 1.2486}
REG_WEIGHTS = {"rf": 0.4671, "xgb": -0.0277, "lgbm": 0.2596, "cb": 0.2698}

def load_data():
    clf_df = pd.read_parquet(f"{DATA_DIR}/classification_ready.parquet")
    reg_df = pd.read_parquet(f"{DATA_DIR}/regression_ready.parquet")
    
    # Feature documentation
    feat_doc = pd.read_csv(f"{DATA_DIR}/feature_documentation.csv")
    features_clf = feat_doc[feat_doc["model"] == "classification"]["feature"].tolist()
    features_reg = feat_doc[feat_doc["model"] == "regression"]["feature"].tolist()
    
    # Temporal Splits (matching 04_Modeling.ipynb)
    TRAIN_MAX = 2017
    VAL_MIN = 2018
    VAL_MAX = 2021
    
    clf_train = clf_df[clf_df['sale_year'] <= TRAIN_MAX].copy()
    clf_val   = clf_df[(clf_df['sale_year'] >= VAL_MIN) & (clf_df['sale_year'] <= VAL_MAX)].copy()
    
    reg_train = reg_df[reg_df['sale_year'] <= TRAIN_MAX].copy()
    reg_val   = reg_df[(reg_df['sale_year'] >= VAL_MIN) & (reg_df['sale_year'] <= VAL_MAX)].copy()
    
    return clf_train, clf_val, reg_train, reg_val, features_clf, features_reg

def train_and_save():
    print("Loading data...")
    clf_train, clf_val, reg_train, reg_val, f_clf, f_reg = load_data()
    
    X_train_c, y_train_c = clf_train[f_clf], clf_train['sold_to_third_party']
    X_val_c, y_val_c = clf_val[f_clf], clf_val['sold_to_third_party']
    
    X_train_r, y_train_r = reg_train[f_reg], reg_train['log_price_gns']
    X_val_r, y_val_r = reg_val[f_reg], reg_val['log_price_gns']

    # --- STAGE 1 (Classification) ---
    print("\nTraining Stage 1 base models...")
    models_c = {}
    
    # Matching the 80/20 split used in notebook for XGB/CB weights
    _split_idx_clf = int(len(X_train_c) * 0.8)
    y_tr_xgb = y_train_c.iloc[:_split_idx_clf]
    _scale_pos = (y_tr_xgb == 0).sum() / max((y_tr_xgb == 1).sum(), 1)

    models_c['rf'] = RandomForestClassifier(**CLF_PARAMS['rf'], random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced")
    models_c['rf'].fit(X_train_c, y_train_c)
    
    models_c['xgb'] = XGBClassifier(**CLF_PARAMS['xgb'], tree_method="hist", device="cpu", n_jobs=-1, scale_pos_weight=_scale_pos, eval_metric="aucpr", random_state=RANDOM_SEED)
    models_c['xgb'].fit(X_train_c, y_train_c)
    
    models_c['lgbm'] = LGBMClassifier(**CLF_PARAMS['lgbm'], random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    models_c['lgbm'].fit(X_train_c, y_train_c)
    
    models_c['cb'] = CatBoostClassifier(**CLF_PARAMS['cb'], loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced", random_seed=RANDOM_SEED, verbose=False, thread_count=-1)
    models_c['cb'].fit(X_train_c, y_train_c)
    
    # Reconstruct Stacking Classifier (using the found weights)
    # Actually, the best way to ensure it matches the audit is to re-fit on Val predictions
    # because we don't have the intercept.
    print("Fitting Stage 1 Meta-learner (LogisticRegression)...")
    Z_val_c = np.column_stack([models_c[k].predict_proba(X_val_c)[:, 1] for k in ["rf", "xgb", "lgbm", "cb"]])
    meta_c = LogisticRegression(random_state=RANDOM_SEED)
    meta_c.fit(Z_val_c, y_val_c)
    print(f"  Coefficients: {meta_c.coef_[0]}")
    print(f"  Intercept: {meta_c.intercept_[0]}")

    # --- STAGE 2 (Regression) ---
    print("\nTraining Stage 2 base models...")
    models_r = {}
    
    models_r['rf'] = RandomForestRegressor(**REG_PARAMS['rf'], random_state=RANDOM_SEED, n_jobs=-1)
    models_r['rf'].fit(X_train_r, y_train_r)
    
    models_r['xgb'] = XGBRegressor(**REG_PARAMS['xgb'], tree_method="hist", device="cpu", n_jobs=-1, objective="reg:squarederror", eval_metric="rmse", random_state=RANDOM_SEED)
    models_r['xgb'].fit(X_train_r, y_train_r)
    
    models_r['lgbm'] = LGBMRegressor(**REG_PARAMS['lgbm'], random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    models_r['lgbm'].fit(X_train_r, y_train_r)
    
    models_r['cb'] = CatBoostRegressor(**REG_PARAMS['cb'], loss_function="RMSE", random_seed=RANDOM_SEED, verbose=False, thread_count=-1)
    models_r['cb'].fit(X_train_r, y_train_r)
    
    print("Fitting Stage 2 Meta-learner (Ridge)...")
    Z_val_r = np.column_stack([models_r[k].predict(X_val_r) for k in ["rf", "xgb", "lgbm", "cb"]])
    meta_r = Ridge(alpha=1.0) # alpha=1.0 was likely the default used in notebook if not optimized
    meta_r.fit(Z_val_r, y_val_r)
    print(f"  Coefficients: {meta_r.coef_}")
    print(f"  Intercept: {meta_r.intercept_}")

    # --- Save artifacts ---
    print("\nSaving models to models/...")
    joblib.dump(models_c, f"{MODELS_DIR}/stage1_base_models.joblib")
    joblib.dump(meta_c, f"{MODELS_DIR}/stage1_meta_learner.joblib")
    joblib.dump(models_r, f"{MODELS_DIR}/stage2_base_models.joblib")
    joblib.dump(meta_r, f"{MODELS_DIR}/stage2_meta_learner.joblib")
    print("Done.")

if __name__ == "__main__":
    train_and_save()
