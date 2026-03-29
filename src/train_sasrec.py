"""Local-first and SageMaker-compatible SASRec training helpers."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

REQUIRED_PIP_SPECS = ["pandas>=2.2", "numpy>=1.26", "pyarrow>=17.0.0"]
for spec in REQUIRED_PIP_SPECS:
    module_name = spec.split(">=")[0]
    try:
        __import__(module_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec])

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .sequence_data import encode_history_tokens, is_low_value_token
except ImportError:
    from sequence_data import encode_history_tokens, is_low_value_token


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PrefixDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, token_to_id: Dict[str, int], max_seq_len: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.token_to_id = token_to_id
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        history_tokens = json.loads(row["prefix_tokens_json"])
        input_ids = encode_history_tokens(history_tokens, self.token_to_id, self.max_seq_len)
        target_id = self.token_to_id.get(str(row["next_token"]), self.token_to_id["<UNK>"])
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)


class SASRecModel(nn.Module):
    def __init__(self, num_items: int, max_seq_len: int, embedding_dim: int = 64, num_blocks: int = 2, num_heads: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, num_items)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        padding_mask = input_ids.eq(0)
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        encoded = self.layer_norm(encoded)
        non_pad_counts = input_ids.ne(0).sum(dim=1).clamp(min=1)
        last_indices = (non_pad_counts - 1).long()
        last_hidden = encoded[torch.arange(batch_size, device=device), last_indices]
        return self.output_layer(last_hidden)


def build_token_maps(train_frame: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab = set(train_frame["next_token"].astype(str).tolist())
    for prefix_json in train_frame["prefix_tokens_json"].astype(str):
        vocab.update(json.loads(prefix_json))
    ordered_vocab = ["<PAD>", "<UNK>"] + sorted(vocab)
    token_to_id = {token: idx for idx, token in enumerate(ordered_vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token


@torch.no_grad()
def evaluate_model(model: SASRecModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    hits_5 = hits_10 = mrr_10 = ndcg_10 = 0.0
    total = 0
    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        logits = model(input_ids)
        _, topk_indices = torch.topk(logits, k=min(10, logits.shape[1]), dim=1)
        for row_idx in range(target_ids.shape[0]):
            total += 1
            target = int(target_ids[row_idx].item())
            ranked = topk_indices[row_idx].tolist()
            if target in ranked[:5]:
                hits_5 += 1.0
            if target in ranked[:10]:
                hits_10 += 1.0
                rank = ranked.index(target) + 1
                mrr_10 += 1.0 / rank
                ndcg_10 += 1.0 / math.log2(rank + 1)
    total = max(total, 1)
    return {
        "HitRate@5": hits_5 / total,
        "HitRate@10": hits_10 / total,
        "MRR@10": mrr_10 / total,
        "NDCG@10": ndcg_10 / total,
    }


def run_sasrec_training(
    sequence_prefix_table: pd.DataFrame,
    sequence_split: Dict[str, Any],
    output_dir: Path | str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = dict(params)
    seed = int(params.get("seed", 42))
    set_seed(seed)

    train_frame = sequence_prefix_table.loc[
        sequence_prefix_table["canonical_user_id"].isin(sequence_split["sequence_train_user_ids"])
    ].copy()
    valid_frame = sequence_prefix_table.loc[
        sequence_prefix_table["canonical_user_id"].isin(sequence_split["sequence_valid_user_ids"])
    ].copy()

    if valid_frame.empty and not train_frame.empty:
        valid_frame = train_frame.sample(n=min(len(train_frame), max(1, int(len(train_frame) * 0.05))), random_state=seed).copy()

    token_to_id, id_to_token = build_token_maps(train_frame)
    low_value_token_ids = [token_to_id[token] for token in token_to_id if token not in {"<PAD>", "<UNK>"} and is_low_value_token(token)]

    max_seq_len = int(params.get("max_seq_len", 50))
    batch_size = int(params.get("batch_size", 256))
    num_workers = int(params.get("num_workers", 0))
    prefer_gpu = bool(params.get("prefer_gpu", False))

    train_dataset = PrefixDataset(train_frame, token_to_id, max_seq_len)
    valid_dataset = PrefixDataset(valid_frame, token_to_id, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    device = torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")
    model = SASRecModel(
        num_items=len(token_to_id),
        max_seq_len=max_seq_len,
        embedding_dim=int(params.get("embedding_dim", 64)),
        num_blocks=int(params.get("num_blocks", 2)),
        num_heads=int(params.get("num_heads", 2)),
        dropout=float(params.get("dropout", 0.2)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss()

    best_metric = -1.0
    best_state = None
    patience_counter = 0
    history = []
    epochs = int(params.get("epochs", 15))
    patience = int(params.get("patience", 3))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, target_ids)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(input_ids.shape[0])

        train_loss = total_loss / max(len(train_dataset), 1)
        valid_metrics = evaluate_model(model, valid_loader, device) if len(valid_dataset) > 0 else {
            "HitRate@5": 0.0,
            "HitRate@10": 0.0,
            "MRR@10": 0.0,
            "NDCG@10": 0.0,
        }
        epoch_row = {"epoch": epoch, "train_loss": train_loss, **valid_metrics}
        history.append(epoch_row)
        print(json.dumps(epoch_row))

        valid_ndcg = valid_metrics["NDCG@10"]
        if valid_ndcg > best_metric:
            best_metric = valid_ndcg
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is None:
        best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    torch.save(
        {
            "model_state_dict": best_state,
            "num_items": len(token_to_id),
            "max_seq_len": max_seq_len,
            "embedding_dim": int(params.get("embedding_dim", 64)),
            "num_blocks": int(params.get("num_blocks", 2)),
            "num_heads": int(params.get("num_heads", 2)),
            "dropout": float(params.get("dropout", 0.2)),
        },
        output_dir / "sasrec_model.pt",
    )
    (output_dir / "token_to_id.json").write_text(json.dumps(token_to_id, indent=2))
    (output_dir / "id_to_token.json").write_text(json.dumps({str(k): v for k, v in id_to_token.items()}, indent=2))
    sequence_config = {
        "model_name": "SASRec",
        "max_seq_len": max_seq_len,
        "embedding_dim": int(params.get("embedding_dim", 64)),
        "num_blocks": int(params.get("num_blocks", 2)),
        "num_heads": int(params.get("num_heads", 2)),
        "dropout": float(params.get("dropout", 0.2)),
        "batch_size": batch_size,
        "lr": float(params.get("lr", 1e-3)),
        "weight_decay": float(params.get("weight_decay", 1e-4)),
        "epochs_requested": epochs,
        "patience": patience,
        "seed": seed,
        "num_workers": num_workers,
        "device": str(device),
        "low_value_token_ids": low_value_token_ids,
        "vocab_size": len(token_to_id),
    }
    sequence_metrics = {"best_validation_metric": best_metric, "history": history}
    (output_dir / "sequence_config.json").write_text(json.dumps(sequence_config, indent=2))
    (output_dir / "sequence_metrics.json").write_text(json.dumps(sequence_metrics, indent=2))
    return {"sequence_config": sequence_config, "sequence_metrics": sequence_metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefer_gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_data_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    model_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    sequence_prefix_table = pd.read_parquet(input_dir / "sequence_prefix_table.parquet")
    sequence_split = json.loads((input_dir / "sequence_split.json").read_text())
    params = {
        "max_seq_len": args.max_seq_len,
        "embedding_dim": args.embedding_dim,
        "num_blocks": args.num_blocks,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "prefer_gpu": bool(args.prefer_gpu),
    }
    result = run_sasrec_training(sequence_prefix_table, sequence_split, model_dir, params)
    (output_data_dir / "sequence_metrics.json").write_text(json.dumps(result["sequence_metrics"], indent=2))
    print(json.dumps(result["sequence_metrics"], indent=2))


if __name__ == "__main__":
    main()
