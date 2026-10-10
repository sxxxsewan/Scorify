import os
import joblib
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

# 1. Пути к файлам
RAW_TEST_PATH = "/content/drive/MyDrive/scorify/data/raw/cs-test.csv"
PROCESSED_TEST_PATH = "/content/drive/MyDrive/scorify/data/processed/test_features.parquet"
MODEL_PATH = "/content/drive/MyDrive/scorify/models/lightgbm_model.pkl"
SUBMISSION_PATH = "/content/drive/MyDrive/scorify/data/processed/submission.csv"

def process_test_data():
    print("Обработка тестовых данных через PySpark...")
    spark = SparkSession.builder \
        .appName("CreditScoringTestPreprocess") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    df = spark.read.csv(RAW_TEST_PATH, header=True, inferSchema=True, nullValue="NA")

    for col in df.columns:
        df = df.withColumnRenamed(col, col.strip())

    if "_c0" in df.columns:
        df = df.drop("_c0")

    for col_name in df.columns:
        df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    df = df \
        .withColumn("MonthlyIncome_isna", F.when(F.col("MonthlyIncome").isNull(), 1).otherwise(0)) \
        .withColumn("NumberOfDependents_isna", F.when(F.col("NumberOfDependents").isNull(), 1).otherwise(0))

    median_deps = df.approxQuantile("NumberOfDependents", [0.5], 0.01)[0]
    if median_deps is None:
        median_deps = 0.0

    df = df.fillna({"NumberOfDependents": median_deps})

    df = df.withColumn("AgeGroup", F.ntile(10).over(Window.orderBy("age")))
    window_age = Window.partitionBy("AgeGroup")
    df = df.withColumn("Median_Income_AgeGroup", F.expr("percentile_approx(MonthlyIncome, 0.5)").over(window_age))
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

    os.makedirs("/content/drive/MyDrive/scorify/data/processed", exist_ok=True)
    df.write.mode("overwrite").parquet(PROCESSED_TEST_PATH)
    print("Тестовые данные успешно обработаны и сохранены")

def make_predictions():
    print("Загрузка модели и подготовленных данных...")
    model = joblib.load(MODEL_PATH)
    df_test = pd.read_parquet(PROCESSED_TEST_PATH)

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

    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Предсказания успешно сохранены в {SUBMISSION_PATH}")

if __name__ == "__main__":
    process_test_data()
    make_predictions()