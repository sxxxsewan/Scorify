from pathlib import Path

import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

RAW_PATH = PROJECT_ROOT / CONFIG["paths"]["raw_train"]
OUTPUT_PATH = PROJECT_ROOT / CONFIG["paths"]["features"]

spark = SparkSession.builder \
    .appName("CreditScoringPreprocess") \
    .config("spark.driver.memory", CONFIG["features"]["spark_driver_memory"]) \
    .getOrCreate()

df = spark.read.csv(
    str(RAW_PATH),
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
    F.ntile(CONFIG["features"]["age_group_count"]).over(Window.orderBy("age")),
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

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.write.mode("overwrite").parquet(str(OUTPUT_PATH))
spark.stop()

print("PySpark Pipeline выполнен успешно! Файл сохранен в Parquet")