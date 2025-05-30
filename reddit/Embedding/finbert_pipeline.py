#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WSB FinBERT Pipeline  ——  Spark Tokenizer (CPU) + FinBERT Embedding (Single GPU)
"""

import os, time, gc, warnings, argparse
import pyarrow.dataset as ds, pyarrow as pa, pyarrow.parquet as pq
import numpy as np, tqdm, torch
from transformers import AutoTokenizer, AutoModel
from pyspark.sql import SparkSession, Row, functions as F, types as T

# ─────────────── Paths ───────────────────────────────────
ROOT   = "/scratch/midway3/zhengzhiyu6689/macs30123/project/reddit"
SRC    = f"{ROOT}/stage03_clean_with_ticker"
TOKOUT = f"{ROOT}/stage03_tok"
EMBOUT = f"{ROOT}/stage04_finbert_emb"
HF_DIR = f"{ROOT}/hf"

# Configuration constants
CPU_THREADS = 16       # Number of CPU threads for Spark
PARTITIONS   = 200     # Number of Spark partitions
SEQ_LEN      = 32      # Maximum token sequence length
BATCH_GPU    = 2048    # Batch size per GPU pass

# ───────────────── Stage-1 : Spark tokenizer ─────────────────
def stage_tokenize():
    """
    Perform text tokenization on cleaned Reddit data using Spark and FinBERT tokenizer.

    1. Initialize a local Spark session with specified CPU_THREADS and PARTITIONS.
    2. Load FinBERT tokenizer from local cache and ensure pad token is set.
    3. Broadcast tokenizer to all Spark executors.
    4. Read cleaned text from Parquet (SRC), repartition, and apply tokenizer in parallel.
    5. Write tokenized output (input_ids and attention_mask) to Parquet (TOKOUT) with Zstandard compression.
    """
    # Suppress HuggingFace hub warnings during tokenization
    warnings.filterwarnings("ignore", module="huggingface_hub")

    # Initialize the Spark session
    spark = (SparkSession.builder
             .appName("FinBERT-TOK")
             .master(f"local[{CPU_THREADS}]")
             .config("spark.sql.shuffle.partitions", PARTITIONS)
             .config("spark.driver.memory", "48g")
             .getOrCreate())
    print("[Stage-1] Spark session started")

    # Load FinBERT tokenizer from offline cache
    tok = AutoTokenizer.from_pretrained(
        "ProsusAI/finbert",
        cache_dir=HF_DIR,
        local_files_only=True,
        use_fast=True
    )
    # Ensure pad token is defined (use EOS token if necessary)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # Broadcast tokenizer to executors
    tok_bc = spark.sparkContext.broadcast(tok)

    # Define schema for tokenized output
    schema = T.StructType([
        T.StructField("id",             T.StringType()),
        T.StructField("ticker",         T.StringType()),
        T.StructField("input_ids",      T.ArrayType(T.ShortType())),
        T.StructField("attention_mask", T.ArrayType(T.ByteType())),
    ])

    # Function to tokenize a batch of rows on each executor
    def tok_part(rows):
        tokenizer = tok_bc.value
        out = []
        for r in rows:
            enc = tokenizer(
                r.clean_text,
                truncation=True,
                max_length=SEQ_LEN,
                padding="max_length"
            )
            out.append(Row(
                id=r.id,
                ticker=r.ticker,
                input_ids=[int(x) for x in enc["input_ids"]],
                attention_mask=[int(x) for x in enc["attention_mask"]]
            ))
        return out

    # Read, tokenize, and write tokenized data to Parquet
    (spark.read.parquet(SRC)
          .repartition(PARTITIONS)
          .rdd.mapPartitions(tok_part)
          .toDF(schema)
          .repartition("ticker")
          .write.mode("overwrite")
          .option("compression", "zstd")
          .parquet(TOKOUT))

    print(f"[Stage-1] ✓ tokenized → {TOKOUT}")
    tok_bc.unpersist()
    spark.stop()
    gc.collect()

# ───────────────── Stage-2 : GPU embedding ───────────────────
def stage_embed():
    """
    Generate FinBERT embeddings from tokenized data on a single GPU.

    1. Load the FinBERT model in half-precision to GPU and set to eval mode.
    2. Read tokenized batches from Parquet (TOKOUT) via pyarrow.dataset.
    3. Accumulate fixed-size batches (BATCH_GPU) in host memory and transfer to GPU.
    4. Extract the [CLS] token embedding (first token) for each input.
    5. Write embeddings to Parquet (EMBOUT) in Zstandard compression.
    """
    print("[Stage-2] Loading FinBERT …")
    # Load pretrained model onto GPU
    model = (AutoModel.from_pretrained(
               "ProsusAI/finbert",
               cache_dir=HF_DIR,
               local_files_only=True,
               torch_dtype=torch.float16)
             .to("cuda:0").eval())

    # Create dataset reader for tokenized Parquet files
    ds_tok = ds.dataset(TOKOUT, format="parquet")
    # Count total rows for progress bar estimation
    total_rows = ds_tok.count_rows()

    # Ensure output directory exists
    os.makedirs(EMBOUT, exist_ok=True)
    writer = None

    # Generator that yields batches of size BATCH_GPU
    def batch_generator():
        ids_buf, mask_buf, mids_buf, tcks_buf = [], [], [], []
        for tbl in ds_tok.to_batches():
            ids   = tbl.column("input_ids").to_pylist()
            masks = tbl.column("attention_mask").to_pylist()
            mids  = tbl.column("id").to_pylist()
            tcks  = tbl.column("ticker").to_pylist()

            # Accumulate rows until batch is full
            for i in range(len(ids)):
                ids_buf.append(ids[i])
                mask_buf.append(masks[i])
                mids_buf.append(mids[i])
                tcks_buf.append(tcks[i])
                if len(ids_buf) == BATCH_GPU:
                    yield (np.stack(ids_buf), np.stack(mask_buf), mids_buf, tcks_buf)
                    ids_buf.clear(); mask_buf.clear()
                    mids_buf.clear(); tcks_buf.clear()

        # Yield any remaining rows
        if ids_buf:
            yield (np.stack(ids_buf), np.stack(mask_buf), mids_buf, tcks_buf)

    # Run embedding with progress bar
    t0 = time.time()
    with torch.inference_mode():
        for ids_np, mask_np, mids, tcks in tqdm.tqdm(
                batch_generator(), total=total_rows // BATCH_GPU + 1):
            # Move data to GPU
            enc_ids  = torch.as_tensor(ids_np,  dtype=torch.int64, device="cuda:0")
            enc_mask = torch.as_tensor(mask_np, dtype=torch.int64, device="cuda:0")

            # Extract [CLS] token embedding
            cls = model(input_ids=enc_ids, attention_mask=enc_mask)
            cls = cls.last_hidden_state[:, 0, :]
            cls = cls.cpu().to(torch.float32).numpy()

            # Build and write Parquet table
            table = pa.Table.from_pydict({
                "id":     pa.array(mids, pa.string()),
                "ticker": pa.array(tcks, pa.string()),
                "embed":  pa.FixedSizeListArray.from_arrays(
                              pa.array(cls.reshape(-1), type=pa.float32()), 768)
            })

            if writer is None:
                # Initialize writer on first batch
                writer = pq.ParquetWriter(
                    os.path.join(EMBOUT, "part-00.parquet"),
                    table.schema, compression="zstd")
            writer.write_table(table)

            # Free GPU memory
            del enc_ids, enc_mask, cls, table
            torch.cuda.empty_cache()

    # Close writer if created
    if writer:
        writer.close()
    print(f"[Stage-2] ✓ embeddings → {EMBOUT} | elapsed {time.time()-t0:,.0f}s")

# ────────────────────────── main ──────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tokenize", action="store_true",
                        help="If stage 1 exists, skip Stage-1")
    args = parser.parse_args()

    # Stage-1: tokenization
    if not args.skip_tokenize:
        stage_tokenize()

    # Clear CUDA cache and run embedding
    torch.cuda.empty_cache()
    stage_embed()
