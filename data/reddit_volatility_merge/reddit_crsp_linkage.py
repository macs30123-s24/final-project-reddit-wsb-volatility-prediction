#!/usr/bin/env python3
"""
author: Zhiyu Zheng

Link Reddit-level observations to the daily CRSP volatility panel by PERMNO and date.
The result is written to a partitioned Parquet dataset (one partition per PERMNO).

Full run:  spark-submit link_reddit_crsp.py
Test run:  spark-submit link_reddit_crsp.py --test
"""

import argparse, itertools, os
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

# ---------- Paths ----------
REDDIT_PATH = "/scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/data/reddit_full1"
CRSP_PATH   = "/scratch/midway3/zhengzhiyu6689/macs30123/project/crsp_raw/CRSP_vol.parquet"
OUT_PATH    = "/scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/data/reddit_crsp"

# ---------- Batch size ----------
BATCH_SIZE = 200
SPARK_LOCAL_DIR = os.environ.get(
    "SPARK_LOCAL_DIRS",
    f"/scratch/midway3/{os.environ['USER']}/spark_local_{os.environ.get('SLURM_JOB_ID','dev')}"
)

# ---------- Utilities ----------
def grouper(it, n):
    """Yield successive chunks of size *n* from iterator *it*."""
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def add_prefix(df, pref):
    """Add a prefix to all column names in a DataFrame."""
    return df.select([F.col(c).alias(f"{pref}_{c}") for c in df.columns])

# ---------- main ----------
def main(test: bool):

    spark = (
        SparkSession.builder
          .appName("reddit_link_crsp")
          .config("spark.local.dir", SPARK_LOCAL_DIR)
          .config("spark.sql.shuffle.partitions", 200)
          .config("spark.network.timeout", "900s")
          .config("spark.executor.heartbeatInterval", "90s")
          .config("spark.shuffle.io.maxRetries", "10")
          .config("spark.shuffle.io.retryWait", "15s")
          .config("spark.shuffle.checksum.enabled", "true")
          .getOrCreate()
    )
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    # 1)  List of PERMNOs present in the Reddit dataset
    all_permnos = (
        spark.read.parquet(REDDIT_PATH)
             .select("PERMNO").distinct()
             .orderBy("PERMNO")
             .rdd.map(lambda r: r.PERMNO).collect()
    )
    if test:
        all_permnos = all_permnos[:3]

    # 2)  Process in batches
    for i, batch in enumerate(grouper(iter(all_permnos), BATCH_SIZE), 1):
        print(f"[Batch {i}] PERMNO {batch[0]} … {batch[-1]} (n={len(batch)})")

        # --- Reddit ---
        df_reddit = (
            spark.read.parquet(REDDIT_PATH)
                 .filter(F.col("PERMNO").isin(batch))
                 .repartition("PERMNO")
        )
        df_reddit = add_prefix(df_reddit, "r")      # Add prefix

        # --- CRSP ---
        df_crsp = (
            spark.read.parquet(CRSP_PATH)
                 .filter(F.col("PERMNO").isin(batch))
                 .repartition("PERMNO")
        )
        df_crsp = add_prefix(df_crsp, "c")

        # --- Join ---
        joined = (
            df_reddit.join(
                df_crsp,
                (F.col("r_PERMNO") == F.col("c_PERMNO")) &
                (F.col("r_created_date") == F.col("c_date")),
                how="inner"            # change to "outer" if you need a full outer join
            )
            .withColumn("PERMNO", F.col("r_PERMNO"))   # write out the partition key
        )

        # --- Append write ---
        (joined.write
               .mode("append")
               .option("compression", "snappy")
               .option("maxRecordsPerFile", 300_000)
               .partitionBy("PERMNO")
               .parquet(OUT_PATH))

        spark.catalog.clearCache()
        print(f"[Batch {i}] written ✓")

    spark.stop()


# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="run only a small test batch")
    args = parser.parse_args()

    Path(SPARK_LOCAL_DIR).mkdir(parents=True, exist_ok=True)
    main(args.test)
