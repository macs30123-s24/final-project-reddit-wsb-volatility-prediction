#!/usr/bin/env python3
"""
author: Zhiyu Zheng

Full run:  spark-submit reddit_full.py  
Test run:  spark-submit reddit_full.py --test  

This Spark job merges two Reddit-derived tables—FinBERT embeddings and
cleaned post metadata—and then uses a Ticker → PERMNO lookup to attach
each post’s CRSP identifier. The result is a row-level dataset ready to
be joined with downstream financial data.  
Run with `--test` to write just three sample rows; otherwise the full
output is hash-partitioned into `N_BUCKETS` Parquet files for efficient
storage and shuffle-friendly reads.

"""

import argparse
import math
from pyspark.sql import SparkSession, functions as F

# ---------- Paths ----------
EMB_PATH   = "/scratch/midway3/zhengzhiyu6689/macs30123/project/reddit/stage04_finbert_emb"
CLEAN_PATH = "/scratch/midway3/zhengzhiyu6689/macs30123/project/reddit/stage03_clean_with_ticker"
MAP_PARQ   = "/scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/data/TICKER_to_PERMNO_v2.parquet"
OUT_PATH   = "/scratch/midway3/zhengzhiyu6689/macs30123/project/prediction/data/reddit_full2"

# ---------- Number of output hash buckets ----------
N_BUCKETS  = 40          # Increase/decrease based on cluster resources


def main(test: bool):
    spark = (
        SparkSession.builder
          .appName("reddit_full_join")
          .config("spark.sql.shuffle.partitions", 200)
          .config("spark.sql.autoBroadcastJoinThreshold", -1)
          .config("spark.network.timeout", "600s")
          .config("spark.executor.heartbeatInterval", "60s")
          .getOrCreate()
    )

    # ① FinBERT embeddings
    df_emb = (
        spark.read.parquet(EMB_PATH)
             .select("id", "embed")
             .coalesce(200)               # one less wide shuffle
             .cache()
    )

    # ② Reddit post metadata
    df_clean = (
        spark.read.parquet(CLEAN_PATH)
             .withColumn("created_ts",   F.from_unixtime("created_utc").cast("timestamp"))
             .withColumn("created_date", F.to_date("created_ts"))
             .coalesce(200)
             .cache()
    )

    # ③ Ticker → PERMNO (small lookup table; safe to broadcast)
    df_map = (
        spark.read.parquet(MAP_PARQ)
             .select(
                 F.to_date("DATE").alias("map_date"),
                 F.col("TICKER").alias("map_ticker"),
                 "PERMNO"
             )
    )
    df_map = F.broadcast(df_map)

    # TEST: sample three rows only
    if test:
        sample_ids = [r.id for r in df_emb.limit(3).collect()]
        df_emb   = df_emb.filter(F.col("id").isin(sample_ids))
        df_clean = df_clean.filter(F.col("id").isin(sample_ids))

    # Join all tables
    joined = (
        df_clean
          .join(df_emb, "id", "left")
          .join(
              df_map,
              (F.col("created_date") == F.col("map_date")) &
              (F.col("ticker")       == F.col("map_ticker")),
              "left"
          )
          .drop("map_date", "map_ticker")
    )

    # Write out 
    if test:
        test_out = OUT_PATH + "_test"
        (joined.write.mode("overwrite")
               .option("compression", "snappy")
               .parquet(test_out))
        joined.select("id", "created_date", "ticker", "PERMNO", "clean_text") \
              .show(3, truncate=120)
        print(f"[TEST] 3 rows saved to {test_out}")

    else:
        # ---------- Loop over buckets for partitioned write ----------
        joined = joined.withColumn("bucket", F.crc32("PERMNO") % N_BUCKETS)
        for i in range(N_BUCKETS):
            (joined.filter(F.col("bucket") == i)
                   .drop("bucket")
                   .write.mode("append")           # append to destination
                   .option("compression", "snappy")
                   .parquet(OUT_PATH))
            spark.catalog.clearCache()            # free executor cache promptly
            print(f"[INFO] bucket {i+1}/{N_BUCKETS} written")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Sample 3 rows, save to <OUT_PATH>_test, and print a small preview",
    )
    args = parser.parse_args()
    main(args.test)
