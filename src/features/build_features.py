import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("CreditScoringPreprocess") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

RAW_PATH = "/content/drive/MyDrive/scorify/data/raw/cs-training.csv"
OUTPUT_PATH = "/content/drive/MyDrive/scorify/data/processed/features.parquet"

df = spark.read.csv(RAW_PATH, header=True, inferSchema=True, nullValue="NA")

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
df.write.mode("overwrite").parquet(OUTPUT_PATH)

print("PySpark Pipeline выполнен успешно! Файл сохранен в Parquet")