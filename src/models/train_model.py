import json
import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)


def main():
    paths = CONFIG["paths"]
    training = CONFIG["training"]
    lightgbm_params = CONFIG["lightgbm"]
    data_path = PROJECT_ROOT / paths["features"]
    model_path = PROJECT_ROOT / paths["model"]
    metrics_path = PROJECT_ROOT / paths["metrics"]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    registry_path = (PROJECT_ROOT / paths["mlflow_registry_db"]).resolve()
    mlflow.set_registry_uri(f"sqlite:///{registry_path.as_posix()}")
    mlflow.set_experiment("scorify-training")

    print("Загрузка обработанных данных из Parquet...")
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=training["test_size"],
        random_state=training["random_state"],
        stratify=y,
    )
    cv = StratifiedKFold(
        n_splits=training["cv_splits"],
        shuffle=True,
        random_state=training["random_state"],
    )
    oof_predictions = np.zeros(len(X_train))
    fold_models = []

    print("Запуск обучения LightGBM с кросс-валидацией...")
    for fold, (train_indices, validation_indices) in enumerate(
        cv.split(X_train, y_train), start=1
    ):
        fold_X_train = X_train.iloc[train_indices]
        fold_y_train = y_train.iloc[train_indices]
        fold_X_validation = X_train.iloc[validation_indices]
        fold_y_validation = y_train.iloc[validation_indices]

        model = lgb.LGBMClassifier(**lightgbm_params)
        model.fit(
            fold_X_train,
            fold_y_train,
            eval_set=[(fold_X_validation, fold_y_validation)],
            callbacks=[
                lgb.early_stopping(
                    training["early_stopping_rounds"], verbose=False
                )
            ],
        )

        validation_predictions = model.predict_proba(fold_X_validation)[:, 1]
        oof_predictions[validation_indices] = validation_predictions
        fold_auc = roc_auc_score(fold_y_validation, validation_predictions)
        fold_models.append((fold_auc, model, fold))
        print(f"Fold {fold} | ROC-AUC: {fold_auc:.5f}")

    oof_auc = roc_auc_score(y_train, oof_predictions)
    best_fold_auc, best_model, best_fold = max(fold_models, key=lambda item: item[0])
    holdout_predictions = best_model.predict_proba(X_holdout)[:, 1]
    holdout_auc = roc_auc_score(y_holdout, holdout_predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    metrics = {
        "roc_auc_oof": float(oof_auc),
        "roc_auc_holdout": float(holdout_auc),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    with mlflow.start_run(run_name="lightgbm-training"):
        mlflow.log_params(
            {f"lightgbm.{key}": value for key, value in lightgbm_params.items()}
        )
        mlflow.log_params(
            {
                "training.test_size": training["test_size"],
                "training.cv_splits": training["cv_splits"],
                "training.random_state": training["random_state"],
                "training.early_stopping_rounds": training[
                    "early_stopping_rounds"
                ],
                "data.feature_path": str(data_path),
                "data.raw_train_path": str(PROJECT_ROOT / paths["raw_train"]),
                "data.model_path": str(model_path),
                "best_fold": best_fold,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path))
        mlflow.lightgbm.log_model(
            lgb_model=best_model.booster_,
            artifact_path="model",
            registered_model_name=training["model_registry_name"],
        )

    print(f"OOF ROC-AUC: {oof_auc:.5f}")
    print(f"Holdout ROC-AUC: {holdout_auc:.5f}")
    print(f"Лучшая модель (Fold {best_fold}, CV ROC-AUC {best_fold_auc:.5f}) сохранена в {model_path}")


if __name__ == "__main__":
    main()