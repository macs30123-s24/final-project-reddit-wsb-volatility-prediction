"""
#!/usr/bin/env python3
author: Zhiyu Zheng

This script loads realized-volatility data from the Reddit–CRSP directory, 
optionally samples it, and produces:
 1) A daily average volatility time series plot,
 2) A scatter‐matrix of volatility and returns,
 3) A 22‐day rolling beta plot.

Example usage:
spark-submit --master local[32] vol_viz.py --sample 0        
spark-submit vol_viz.py --sample 50000              
"""
#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
from pyspark.sql import SparkSession, functions as F, Window
import matplotlib
matplotlib.use("Agg")          # No GUI available on server
import matplotlib.pyplot as plt
import pandas as pd

# ───────── Paths ─────────
DATA_DIR = (
    "/scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/data/reddit_crsp"
)  # Read the entire directory; Spark will recurse through all PERMNO=* folders
FIG_DIR = (
    "/scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/fig/amd"
)
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

# ───────── Command-line Interface ─────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--sample",
    type=int,
    default=200_000,
    help="Maximum number of rows to convert to pandas; 0 means full dataset",
)
args = parser.parse_args()
SAMPLE_ROWS = args.sample

# ───────── Spark Configuration ─────────
spark = (
    SparkSession.builder
    .appName("volatility_viz")
    .config("spark.sql.shuffle.partitions", 200)
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")  # Disable Arrow to avoid JDK conflicts
    .getOrCreate()
)

# ───────── 1) Load & Sample Data ─────────
df = (
    spark.read.parquet(DATA_DIR)
    .select(
        # Select only the columns we need and rename them
        F.col("c_date").alias("date"),
        F.col("c_vol_1").alias("vol_1"),
        F.col("c_vol_5").alias("vol_5"),
        F.col("c_vol_22").alias("vol_22"),
        F.col("c_vol_63").alias("vol_63"),
        F.col("c_y_1").alias("y_1"),
    )
    .filter(F.col("date").isNotNull())
)

if SAMPLE_ROWS > 0:
    df = df.orderBy(F.rand(seed=42)).limit(SAMPLE_ROWS)

df.persist()
print(f"Loaded {df.count():,} rows for plotting", file=sys.stderr)

# ───────── 2) Time Series Line Plot ─────────
ts = (
    df.groupBy("date")
    .agg(
        F.avg("vol_1").alias("v1"),
        F.avg("vol_5").alias("v5"),
        F.avg("vol_22").alias("v22"),
        F.avg("vol_63").alias("v63"),
    )
    .orderBy("date")
    .toPandas()
)

plt.figure(figsize=(10, 4))
plt.plot(ts["date"], ts["v1"], label="1 d")
plt.plot(ts["date"], ts["v5"], label="5 d")
plt.plot(ts["date"], ts["v22"], label="1 m")
plt.plot(ts["date"], ts["v63"], label="3 m")
plt.title("Realized Volatility – Daily Average")
plt.ylabel("Volatility")
plt.legend(ncol=4)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/vol_timeseries.png", dpi=150)
plt.close()

# ───────── 3) Scatter Matrix ─────────
pair_pdf = df.orderBy(F.rand()).limit(5_000).toPandas()
pd.plotting.scatter_matrix(
    pair_pdf[["vol_1", "vol_5", "vol_22", "vol_63", "y_1"]],
    diagonal="kde",
    figsize=(8, 8),
    alpha=0.3,
    s=10,
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/scatter_matrix.png", dpi=150)
plt.close()

# ───────── 4) 22-day Rolling Beta ─────────
w = Window.orderBy("date").rowsBetween(-21, 0)
beta_df = (
    df.select("date", "vol_1", "y_1")
    .withColumn("cov", F.covar_pop("vol_1", "y_1").over(w))
    .withColumn("var", F.var_pop("vol_1").over(w))
    .withColumn("beta", F.col("cov") / F.col("var"))
    .orderBy("date")
    .select("date", "beta")
    .toPandas()
)

plt.figure(figsize=(10, 3))
plt.plot(beta_df["date"], beta_df["beta"])
plt.axhline(0, lw=0.5)
plt.title("22-day Rolling β")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/beta_rolling.png", dpi=150)
plt.close()

spark.stop()
print("Figures saved to:", FIG_DIR)
