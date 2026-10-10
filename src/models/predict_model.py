from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

def process_test_data():
    paths = CONFIG["paths"]
    print("Обработка тестовых данных через PySpark...")
    spark = SparkSession.builder \
        .appName("CreditScoringTestPreprocess") \
        .config("spark.driver.memory", CONFIG["features"]["spark_driver_memory"]) \
        .getOrCreate()

    df = spark.read.csv(
        str(PROJECT_ROOT / paths["raw_test"]),
        header=True,
        inferSchema=True,
        nullValue=CONFIG["features"]["null_value"],
    )

    for col in df.columns:
        df = df.withColumnRenamed(col, col.strip())

    if "_c0" in df.columns:
        df = df.drop("_c0")

    for col_name in df.columns:
        df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    df = df \
        .withColumn("MonthlyIncome_isna", F.when(F.col("MonthlyIncome").isNull(), 1).otherwise(0)) \
        .withColumn("NumberOfDependents_isna", F.when(F.col("NumberOfDependents").isNull(), 1).otherwise(0))

    median_deps = df.approxQuantile(
        "NumberOfDependents",
        [CONFIG["features"]["dependents_median_quantile"]],
        CONFIG["features"]["dependents_quantile_error"],
    )[0]
    if median_deps is None:
        median_deps = 0.0

    df = df.fillna({"NumberOfDependents": median_deps})

    df = df.withColumn(
        "AgeGroup",
        F.ntile(CONFIG["features"]["age_group_count"]).over(
            Window.orderBy("age")
        ),
    )
    window_age = Window.partitionBy("AgeGroup")
    income_median_quantile = CONFIG["features"]["income_median_quantile"]
    df = df.withColumn(
        "Median_Income_AgeGroup",
        F.expr(f"percentile_approx(MonthlyIncome, {income_median_quantile})").over(
            window_age
        ),
    )
    df = df.withColumn("MonthlyIncome", F.coalesce(F.col("MonthlyIncome"), F.col("Median_Income_AgeGroup"))).drop("AgeGroup", "Median_Income_AgeGroup")

    df = df \
        .withColumn("Total_PastDue_Count", 
                    F.col("NumberOfTime30-59DaysPastDueNotWorse") + 
                    F.col("NumberOfTime60-89DaysPastDueNotWorse") + 
                    F.col("NumberOfTimes90DaysLate")) \
        .withColumn("Severe_PastDue_Ratio", 
                    F.col("NumberOfTimes90DaysLate") / (F.col("Total_PastDue_Count") + 1)) \
        .withColumn("Income_Per_Person", 
                    F.col("MonthlyIncome") / (F.col("NumberOfDependents") + 1)) \
        .withColumn("Total_Credit_Lines", 
                    F.col("NumberOfOpenCreditLinesAndLoans") + F.col("NumberRealEstateLoansOrLines")) \
        .withColumn("Monthly_Debt_Amount", 
                    F.col("MonthlyIncome") * F.col("DebtRatio"))

    processed_test_path = PROJECT_ROOT / paths["test_features"]
    processed_test_path.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(str(processed_test_path))
    spark.stop()
    print("Тестовые данные успешно обработаны и сохранены")

def make_predictions():
    paths = CONFIG["paths"]
    print("Загрузка модели и подготовленных данных...")
    model = joblib.load(PROJECT_ROOT / paths["model"])
    df_test = pd.read_parquet(PROJECT_ROOT / paths["test_features"])

    if 'SeriousDlqin2yrs' in df_test.columns:
        X_test = df_test.drop(columns=['SeriousDlqin2yrs'])
    else:
        X_test = df_test

    print("Генерация предсказаний...")
    preds = model.predict_proba(X_test)[:, 1]

    sub = pd.DataFrame({
        'Id': np.arange(1, len(preds) + 1),
        'Probability': preds
    })

    submission_path = PROJECT_ROOT / paths["submission"]
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(submission_path, index=False)
    print(f"Предсказания успешно сохранены в {submission_path}")

if __name__ == "__main__":
    process_test_data()
    make_predictions()