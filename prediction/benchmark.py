#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Zhiyu
HAR-style rolling forecast:
    • target      : next-day volatility  (y_1)
    • regressors  : σ(1d), σ(5d), σ(22d), σ(63d)
    • scheme      : expanding 4-year window → 1-year test
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, year as spark_year
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# 0) Spark session ─ launched via spark-submit; master string comes from Slurm
spark = (
    SparkSession.builder
        .appName("HAR-4y-1y")
        .getOrCreate()
)
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# 1) Load CRSP-level volatility features + target
DATA_PATH = (
    "/scratch/midway3/zhengzhiyu6689/macs30123/project/"
    "crsp_raw/CRSP_vol.parquet"
)

df = (
    spark.read.parquet(DATA_PATH)
        .select(
            "PERMNO", "date", "y_1",
            "vol_1", "vol_5", "vol_22", "vol_63",
        )
        .na.drop()                         # drop any missing cells
        .withColumn("year", spark_year("date"))
        .cache()
)
years = sorted(r.year for r in df.select("year").distinct().collect())

# 2) Common plumbing: feature builder, model, and evaluators

assembler = VectorAssembler(
    inputCols=["vol_1", "vol_5", "vol_22", "vol_63"],
    outputCol="features",
)
lr = LinearRegression(
    featuresCol="features",
    labelCol="y_1",
    fitIntercept=True,
    elasticNetParam=0.0,   # plain OLS
    regParam=0.0,
    solver="normal",
)
evaluator_mse = RegressionEvaluator(
    labelCol="y_1", predictionCol="prediction", metricName="mse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="y_1", predictionCol="prediction", metricName="r2"
)

per_year_stats = []

# 3) Walk forward: 4-year training window → next-year test
for test_year in years:
    train_years = [y for y in years if test_year - 4 <= y <= test_year - 1]
    if len(train_years) < 4:
        continue  # not enough history yet

    train_df = df.filter(col("year").isin(train_years))
    test_df = df.filter(col("year") == test_year)

    if train_df.rdd.isEmpty() or test_df.rdd.isEmpty():
        continue

    model = lr.fit(assembler.transform(train_df))
    pred = model.transform(assembler.transform(test_df)).cache()

    mse = evaluator_mse.evaluate(pred)
    r2 = evaluator_r2.evaluate(pred)

    per_year_stats.append((test_year, mse, r2))
    print(f"[{test_year}]  MSE={mse:,.6f}   R²={r2:,.4f}")

# 4) Aggregate performance using the entire evaluation period

first_eval_year = min(y for y, _, _ in per_year_stats) + 4
all_test_df = df.filter(col("year") >= first_eval_year)
pred_all = assembler.transform(all_test_df)

model_all = lr.fit(pred_all)               # refit on full training span
predicted = model_all.transform(pred_all).cache()

total_mse = evaluator_mse.evaluate(predicted)
total_r2 = evaluator_r2.evaluate(predicted)

print("\n===========  SUMMARY  ===========")
for y, mse, r2 in per_year_stats:
    print(f"{y}:  MSE={mse:,.6f}   R²={r2:,.4f}")
print("---------------------------------")
print(f"TOTAL  MSE={total_mse:,.6f}   R²={total_r2:,.4f}")

spark.stop()
