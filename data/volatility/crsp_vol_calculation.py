
"""
crsp_vol_calculation
author: Zhiyu Zheng

Build daily realised-volatility panels from raw CRSP returns
------------------------------------------------------------------------
• Reads the big CRSP.csv   file (RET / RETX / DLRET) from disk.
• Cleans those three return columns into one numeric 'ret'.
• Repartitions by PERMNO so every stock lives in a single partition.
• Computes rolling RV over 1, 5, 22, 63 trading-day windows.
• Creates one-day-ahead targets  y_1 / y_5 / y_22 / y_63.
• Writes the result as Snappy-compressed Parquet, partitioned by PERMNO,
  so later jobs can open a single stock instantly.

Run with: run_vol_calculation.sbatch
"""


import os
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, when, regexp_replace, to_date,
    sum as _sum, sqrt, lead
)

# ---------------------------------------------------------------------
# 0) Build a Spark session
# ---------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("CRSP-Vol")
    .getOrCreate()
)

# ---------------------------------------------------------------------
# 1) Read the raw CRSP CSV (keep only the columns we need)
# ---------------------------------------------------------------------
csv_path = (
    "/scratch/midway3/zhengzhiyu6689/macs30123/project/"
    "crsp_raw/CRSP.csv"
)

df = (
    spark.read.csv(csv_path, header=True)
        .select("PERMNO", "date", "RET", "RETX", "DLRET")
        .withColumn(
            "date",
            to_date(col("date").cast("string"), "yyyyMMdd")
        )
)

# ---------------------------------------------------------------------
# 2) Collapse RET / RETX / DLRET into a single numeric return column
# ---------------------------------------------------------------------
def str_to_num(c: str):
    """Strip single-letter flags (e.g. 'C', 'B') and cast to double."""
    return regexp_replace(col(c), r"^[A-Z]$", "").cast("double")


df = (
    df.withColumn("ret", str_to_num("RET"))
      .withColumn(
          "ret",
          when(col("ret").isNull(), str_to_num("RETX"))
          .otherwise(col("ret"))
      )
      .withColumn(
          "ret",
          when(col("ret").isNull(), str_to_num("DLRET"))
          .otherwise(col("ret"))
      )
      .select("PERMNO", "date", "ret")
)

# ---------------------------------------------------------------------
# 3) Repartition by PERMNO so window ops stay node-local
# ---------------------------------------------------------------------
df = df.repartition("PERMNO")

# ---------------------------------------------------------------------
# 4) Rolling realised volatility for 1 / 5 / 22 / 63 trading-day windows
# ---------------------------------------------------------------------
df = df.withColumn("ret_sq", col("ret") * col("ret"))

for w in (1, 5, 22, 63):
    win = (
        Window.partitionBy("PERMNO")
              .orderBy("date")
              .rowsBetween(-(w - 1), 0)      # inclusive window: t-w+1 … t
    )
    df = df.withColumn(f"vol_{w}", sqrt(_sum("ret_sq").over(win)))

df = df.drop("ret_sq")

# ---------------------------------------------------------------------
# 5) One-day-ahead targets y_1 / y_5 / y_22 / y_63 for forecasting
# ---------------------------------------------------------------------
order_win = Window.partitionBy("PERMNO").orderBy("date")
for w in (1, 5, 22, 63):
    df = df.withColumn(f"y_{w}", lead(f"vol_{w}", 1).over(order_win))

# ---------------------------------------------------------------------
# 6) Write result to Parquet, partitioned by PERMNO
# ---------------------------------------------------------------------
out_path = (
    "/scratch/midway3/zhengzhiyu6689/macs30123/project/"
    "crsp_raw/CRSP_vol.parquet"
)

(df
 .sort("PERMNO", "date")            # optional: sorted output for readability
 .write.mode("overwrite")
 .partitionBy("PERMNO")             # single-stock files open instantly
 .parquet(out_path)
)

spark.stop()
print("saved", out_path)
