#!/usr/bin/env python3
"""
embadding_model.py

author: Zhiyu Zheng

Train a rolling-window ridge-regression model that combines Reddit embeddings
with CRSP volatility features.  Cross-validation selects λ in each window.

Example dry run (20 % sample):  spark-submit reddit_ridge_roll.py --sample 0.2
                 full sample :  spark-submit reddit_ridge_roll.py 

"""

import argparse, datetime as dt
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.functions import array_to_vector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ───────── Paths ─────────
PARQ_DIR  = "/scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/data/reddit_crsp"
MODEL_DIR = "/scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/model/reddit_ridge_ful_amd"

# ───────── CLI ─────────
parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=float, default=1.0, help="sample fraction 0-1")
parser.add_argument("--start-year", type=int, default=2012)
parser.add_argument("--end-year",   type=int, default=2022)
args = parser.parse_args()

# ───────── Spark ─────────
spark = (
    SparkSession.builder
      .appName("ridge_rolling")
      .config("spark.sql.parquet.enableVectorizedReader", "false")  # avoid embed OOM
      .getOrCreate()
)

# ───────── 1) Load / Pre-process ─────────
df = spark.read.parquet(PARQ_DIR)
if args.sample < 1.0:
    df = df.sample(False, args.sample, seed=2025)

df = (
    df
    # Standardize date column (use CRSP's c_date)
    .withColumn("date", F.col("c_date").cast("date"))
    # Reddit embedding vector
    .withColumn("embed_vec", array_to_vector("r_embed"))
    # Keep only rows without missing values
    .filter(
        F.col("c_y_1").isNotNull() &
        F.col("c_vol_1").isNotNull() &
        F.col("c_vol_5").isNotNull() &
        F.col("c_vol_22").isNotNull() &
        F.col("c_vol_63").isNotNull()
    )
)

assembler = VectorAssembler(
    inputCols=["c_vol_1", "c_vol_5", "c_vol_22", "c_vol_63", "embed_vec"],
    outputCol="features")
df = assembler.transform(df).cache()

# ───────── 2) Evaluators ─────────
evaluator_mse = RegressionEvaluator(labelCol="c_y_1", predictionCol="prediction", metricName="mse")
evaluator_r2  = RegressionEvaluator(labelCol="c_y_1", predictionCol="prediction", metricName="r2")

# ───────── 3) Ridge + CV ─────────
ridge = LinearRegression(featuresCol="features", labelCol="c_y_1", elasticNetParam=0.0)
paramGrid = ParamGridBuilder().addGrid(ridge.regParam, [1e-4, 1e-3, 1e-2, 1e-1, 1.0]).build()
cv = CrossValidator(estimator=ridge, estimatorParamMaps=paramGrid,
                    evaluator=evaluator_mse, numFolds=3, parallelism=8)

# ───────── 4) Rolling windows ─────────
results, best = [], None
for y in range(args.start_year, args.end_year - 4):
    train_start, train_end = dt.date(y,1,1),            dt.date(y+3,1,1) - dt.timedelta(days=1)
    test_start,  test_end  = dt.date(y+4,1,1),          dt.date(y+5,1,1) - dt.timedelta(days=1)

    train = df.filter((F.col("date") >= train_start) & (F.col("date") <= train_end))
    test  = df.filter((F.col("date") >= test_start)  & (F.col("date") <= test_end))
    if train.count() < 100 or test.count() == 0:
        print(f"{y}-{y+5} skipped (too few rows)"); continue

    cv_model = cv.fit(train)
    best = cv_model.bestModel
    lam  = best._java_obj.getRegParam()

    pred = best.transform(test)
    mse  = evaluator_mse.evaluate(pred)
    r2   = evaluator_r2.evaluate(pred)
    results.append((y, lam, mse, r2))
    print(f"{y}-{y+5}  λ={lam:.4g}  MSE={mse:.5f}  R²={r2:.4f}")

# ───────── 5) Summary & Save ─────────
if results:
    avg_mse = sum(r[2] for r in results) / len(results)
    avg_r2  = sum(r[3] for r in results) / len(results)
    print(f"\n Avg over {len(results)} windows:  MSE={avg_mse:.5f}  R²={avg_r2:.4f}")

if best:
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    best.write().overwrite().save(MODEL_DIR)
    print("Last-window model saved to", MODEL_DIR)

spark. Stop()
