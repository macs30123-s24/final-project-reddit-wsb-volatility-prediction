#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WSB FinBERT end-to-end pipeline
  • Stage 1: CPU-parallel tokenization in Spark (local[N] threads)
  • Stage 2: GPU embeddings
            – multi-GPU via torch.multiprocessing.spawn if ≥2 GPUs
            – single-GPU fallback otherwise

This script orchestrates tokenization of Reddit WSB comments using FinBERT's tokenizer
and then generates embeddings using PyTorch on available GPUs, writing output to Parquet.
"""

import os
import gc
import warnings
import argparse
import datetime
import shutil
import pathlib

import torch
import torch.multiprocessing as mp
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import tqdm
from transformers import AutoTokenizer, AutoModel
from pyspark.sql import SparkSession, Row, functions as F, types as T

# ─────────────────────────── paths & constants ────────────────────────────
ROOT   = "/scratch/midway3/zhengzhiyu6689/macs30123/project/reddit"
SRC    = f"{ROOT}/stage03_clean_with_ticker"          # cleaned Reddit WSB data
TOKOUT = f"{ROOT}/stage03_tok"                        # tokenized output
EMBOUT = f"{ROOT}/stage04_finbert_emb_multigpu"       # embeddings output
HF_DIR = f"{ROOT}/hf"                                 # offline HF cache

CPU_THREADS = 16   # number of CPU threads for Spark tokenization
PARTITIONS  = 200  # number of partitions for Spark and Parquet
SEQ_LEN     = 64   # maximum sequence length for tokenizer
BATCH_GPU   = 2048 # batch size per GPU

# ─────────────────────────── Stage 1: tokenization ────────────────────────
def stage_tokenize():
    """
    Tokenize text with Spark on CPU.

    Reads cleaned text from Parquet, partitions it across Spark executors,
    applies the FinBERT tokenizer in parallel, and writes tokenized output
    back to Parquet with Zstandard compression.
    """
    # Suppress irrelevant warnings from HuggingFace
    warnings.filterwarnings("ignore", module="huggingface_hub")

    # Initialize Spark session
    spark = (SparkSession.builder
             .appName("FinBERT-TOK")
             .master(f"local[{CPU_THREADS}]")
             .config("spark.sql.shuffle.partitions", PARTITIONS)
             .config("spark.driver.memory", "48g")
             .getOrCreate())
    print(f"[Stage-1] Spark started @ {datetime.datetime.now()}")

    # Load the FinBERT tokenizer from local cache
    tok = AutoTokenizer.from_pretrained(
        "ProsusAI/finbert",
        cache_dir=HF_DIR,
        local_files_only=True,
        use_fast=True
    )
    # Ensure pad token is defined, reuse EOS if necessary
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    # Broadcast tokenizer to executors
    tok_bc = spark.sparkContext.broadcast(tok)

    # Define output schema for tokenized data
    schema = T.StructType([
        T.StructField("id",             T.StringType()),
        T.StructField("ticker",         T.StringType()),
        T.StructField("input_ids",      T.ArrayType(T.ShortType())),
        T.StructField("attention_mask", T.ArrayType(T.ByteType())),
    ])

    def tok_part(rows):
        """
        Tokenize a Spark partition of rows.

        Each row contains 'clean_text', and this function applies the tokenizer
        and returns rows of (id, ticker, input_ids, attention_mask).
        """
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

    # Read cleaned data, tokenize, and write tokenized Parquet
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

# ───────────────────── Stage 2: multi-GPU embedding ───────────────────────
def _embed_worker(rank: int, world: int, tok_path: str, out_path: str):
    """
    Worker function for multi-GPU embedding.

    Each subprocess binds to a specific GPU (by rank), reads its share of
    tokenized batches in round-robin fashion, runs the FinBERT model to extract
    CLS embeddings, and writes them to Parquet.

    Args:
        rank (int): rank of this worker (GPU index)
        world (int): total number of GPUs/processes
        tok_path (str): path to tokenized Parquet dataset
        out_path (str): directory to write embedding Parquet files
    """
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    print(f"[R{rank}] {torch.cuda.get_device_name(rank)} | pid={os.getpid()}")

    # Load pretrained FinBERT model in half precision
    model = (AutoModel.from_pretrained(
                "ProsusAI/finbert",
                cache_dir=HF_DIR,
                local_files_only=True,
                torch_dtype=torch.float16)
             .to(device).eval())

    # Prepare Parquet writer for this rank
    writer = pq.ParquetWriter(
        os.path.join(out_path, f"part-{rank:02d}.parquet"),
        pa.schema([
            ("id",     pa.string()),
            ("ticker", pa.string()),
            ("embed",  pa.list_(pa.float32(), 768))
        ]),
        compression="zstd"
    )

    # Create dataset reader
    ds_tok = ds.dataset(tok_path, format="parquet")
    batch_idx = -1

    with torch.inference_mode():
        for tbl in ds_tok.to_batches():
            batch_idx += 1
            # Round-robin assignment of batches to ranks
            if batch_idx % world != rank:
                continue

            ids   = tbl.column("input_ids").to_pylist()
            masks = tbl.column("attention_mask").to_pylist()
            mids  = tbl.column("id").to_pylist()
            tcks  = tbl.column("ticker").to_pylist()

            # Process in GPU-friendly chunks
            for s in range(0, len(ids), BATCH_GPU):
                enc_ids  = torch.as_tensor(
                    np.stack(ids[s:s+BATCH_GPU]),
                    dtype=torch.int64,
                    device=device
                )
                enc_mask = torch.as_tensor(
                    np.stack(masks[s:s+BATCH_GPU]),
                    dtype=torch.int64,
                    device=device
                )
                # Extract CLS token embeddings
                cls = model(
                        input_ids=enc_ids,
                        attention_mask=enc_mask
                      ).last_hidden_state[:, 0, :]
                cls = cls.cpu().to(torch.float32).numpy()

                # Write embedding batch to Parquet
                table = pa.Table.from_pydict({
                    "id":     pa.array(mids[s:s+BATCH_GPU],  pa.string()),
                    "ticker": pa.array(tcks[s:s+BATCH_GPU], pa.string()),
                    "embed":  pa.FixedSizeListArray.from_arrays(
                                  pa.array(cls.reshape(-1), type=pa.float32()),
                                  768)
                })
                writer.write_table(table)

    writer.close()
    print(f"[R{rank}] finished.")


def stage_embed_multi():
    """
    Orchestrate multi-GPU embedding stage.

    Spawns one process per GPU, each running `_embed_worker`. Ensures output
    directory is clean before starting.
    """
    world = torch.cuda.device_count()
    print(f"[Stage-2] Multi-GPU world = {world}")

    # Prepare output directory
    out_dir = pathlib.Path(EMBOUT)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Launch workers
    mp.spawn(
        _embed_worker,
        args=(world, TOKOUT, EMBOUT),
        nprocs=world,
        join=True
    )
    print("[Stage-2] ✓ embeddings →", EMBOUT)

# ───────────────────── Stage 2: single-GPU fallback ───────────────────────
def stage_embed_single():
    """
    Single GPU embedding (no multiprocessing).

    Reads all tokenized batches sequentially and writes embeddings to a single
    Parquet file.
    """
    print("[Stage-2] Single GPU mode")
    model = (AutoModel.from_pretrained(
                "ProsusAI/finbert",
                cache_dir=HF_DIR,
                local_files_only=True,
                torch_dtype=torch.float16)
             .to("cuda:0").eval())

    ds_tok = ds.dataset(TOKOUT, format="parquet")
    writer = pq.ParquetWriter(
        os.path.join(EMBOUT, "part-00.parquet"),
        pa.schema([
            ("id",     pa.string()),
            ("ticker", pa.string()),
            ("embed",  pa.list_(pa.float32(), 768))
        ]),
        compression="zstd"
    )

    def gen_batches():
        """
        Generate fixed-size batches from tokenized dataset without exceeding memory.

        Yields tuples of (ids, masks, mids, tcks) for each batch.
        """
        ids, masks, mids, tcks = [], [], [], []
        for tbl in ds_tok.to_batches():
            ids  .extend(tbl.column("input_ids").to_pylist())
            masks.extend(tbl.column("attention_mask").to_pylist())
            mids .extend(tbl.column("id").to_pylist())
            tcks.extend(tbl.column("ticker").to_pylist())
            while len(ids) >= BATCH_GPU:
                yield (ids[:BATCH_GPU], masks[:BATCH_GPU],
                       mids[:BATCH_GPU], tcks[:BATCH_GPU])
                del ids[:BATCH_GPU], masks[:BATCH_GPU], mids[:BATCH_GPU], tcks[:BATCH_GPU]
        if ids:
            yield ids, masks, mids, tcks

    with torch.inference_mode():
        for ids_l, masks_l, mids_l, tcks_l in tqdm.tqdm(gen_batches()):
            enc_ids  = torch.as_tensor(
                np.stack(ids_l),
                dtype=torch.int64,
                device="cuda:0"
            )
            enc_mask = torch.as_tensor(
                np.stack(masks_l),
                dtype=torch.int64,
                device="cuda:0"
            )
            cls = model(
                    input_ids=enc_ids,
                    attention_mask=enc_mask
                  ).last_hidden_state[:, 0, :]
            cls = cls.cpu().to(torch.float32).numpy()
            table = pa.Table.from_pydict({
                "id":     pa.array(mids_l,  pa.string()),
                "ticker": pa.array(tcks_l, pa.string()),
                "embed":  pa.FixedSizeListArray.from_arrays(
                              pa.array(cls.reshape(-1), type=pa.float32()),
                              768)
            })
            writer.write_table(table)
    writer.close()
    print("[Stage-2] ✓ embeddings →", EMBOUT)

# ──────────────────────────────── main ────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FinBERT tokenization and embedding pipeline.")
    parser.add_argument("--skip-tokenize", action="store_true",
                        help="Skip Stage-1 tokenization if TOKOUT already exists")
    args = parser.parse_args()

    # Stage-1: tokenization
    if not args.skip_tokenize:
        stage_tokenize()

    # Clear any leftover CUDA cache
    torch.cuda.empty_cache()
    # Stage-2: embedding (multi- or single-GPU)
    if torch.cuda.device_count() > 1:
        stage_embed_multi()
    else:
        stage_embed_single()

