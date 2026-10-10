import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA_PATH = "/content/drive/MyDrive/scorify/data/processed/features.parquet"
MODEL_DIR = "/content/drive/MyDrive/scorify/models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Загрузка обработанных данных из Parquet...")
df = pd.read_parquet(DATA_PATH)

X = df.drop(columns=['SeriousDlqin2yrs'])
y = df['SeriousDlqin2yrs']

print(f"Размерность признаков: {X.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(df))
models = []

print("Запуск обучения LightGBM (5-Fold Stratified K-Fold)...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    fold_auc = roc_auc_score(y_val, val_preds)
    models.append(model)
    
    print(f"Fold {fold + 1} | ROC-AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"Итоговый OOF ROC-AUC: {overall_auc:.5f}")

best_fold_idx = np.argmax([
    roc_auc_score(y.iloc[val_idx], model.predict_proba(X.iloc[val_idx])[:, 1]) 
    for _, val_idx in skf.split(X, y)
])
best_model = models[best_fold_idx]

model_path = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
joblib.dump(best_model, model_path)
print(f"Лучшая модель (Fold {best_fold_idx + 1}) успешно сохранена в {model_path}")