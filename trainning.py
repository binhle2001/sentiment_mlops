# train_bge_m3_trainer.py
"""
Fine-tune BGE-M3 embeddings using HuggingFace Trainer.
- Load your triplet/jsonl: DATA_PATH = "embedding_training_data.jsonl"
- Supports USE_MULTIPLE_NEG (InfoNCE) or Triplet.
- Uses custom data_collator that stacks already-padded tensors from dataset,
  avoiding tokenizer.pad() errors.
"""

import os
import json
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer


# ========== CONFIG ==========
DATA_PATH = "/kaggle/input/fffffff/embedding_training_data.jsonl"
MODEL_NAME = "BAAI/bge-m3"
OUTPUT_DIR = "./bge-m3-finetuned-transformer"
USE_MULTIPLE_NEG = True      # True -> InfoNCE (batch negatives). False -> Triplet (requires neg)
MAX_LENGTH = 128
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 3
LR = 2e-5
FP16 = True                  # mixed precision
LOGGING_STEPS = 50
SAVE_STEPS = 500
SCALE = 20.0                 # scale factor for logits (InfoNCE)
MARGIN = 0.2                 # margin for triplet loss
SEED = 42
# ============================


def load_jsonl(path: str) -> List[Dict[str, Optional[str]]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = obj.get("query") or obj.get("Query") or obj.get("input")
            p = obj.get("pos") or obj.get("positive")
            n = obj.get("neg") or obj.get("negative")
            if q and p:
                items.append({"query": str(q), "pos": str(p), "neg": str(n) if n else None})
    return items


class JsonlContrastiveDataset(Dataset):
    """
    Trả về tensor đã pad (padding="max_length") — phù hợp với data_collator simple stacking.
    Keys returned:
      - query_input_ids, query_attention_mask
      - pos_input_ids, pos_attention_mask
      - optionally neg_input_ids, neg_attention_mask (if use_neg True and neg present)
    """
    def __init__(self, items: List[Dict], tokenizer: AutoTokenizer, max_length: int = 128, use_neg: bool = False):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_neg = use_neg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        q = it["query"]
        p = it["pos"]

        enc_q = self.tokenizer(q, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        enc_p = self.tokenizer(p, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        out = {
            "query_input_ids": enc_q["input_ids"].squeeze(0),
            "query_attention_mask": enc_q["attention_mask"].squeeze(0),
            "pos_input_ids": enc_p["input_ids"].squeeze(0),
            "pos_attention_mask": enc_p["attention_mask"].squeeze(0),
        }

        if self.use_neg:
            n = it.get("neg")
            if n:
                enc_n = self.tokenizer(n, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
                out.update({
                    "neg_input_ids": enc_n["input_ids"].squeeze(0),
                    "neg_attention_mask": enc_n["attention_mask"].squeeze(0),
                })
            else:
                # Fallback: zero tensors (should be avoided ideally)
                out.update({
                    "neg_input_ids": torch.zeros_like(out["pos_input_ids"]),
                    "neg_attention_mask": torch.zeros_like(out["pos_attention_mask"]),
                })
        return out


def my_data_collator(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Simple collator that stacks tensors for each key.
    Expects each feature to already contain padded tensors of same shape.
    """
    batch = {}
    for k in features[0].keys():
        batch[k] = torch.stack([f[k] for f in features], dim=0)
    return batch


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state: (B, L, D), attention_mask: (B, L)
    returns (B, D) mean-pooled over non-masked positions.
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def compute_contrastive_loss(model, inputs: Dict[str, torch.Tensor], use_multiple_neg: bool = True,
                             scale: float = 20.0, margin: float = 0.2) -> torch.Tensor:
    """
    inputs: dict of tensors already moved to correct device by Trainer.
    """
    device = next(model.parameters()).device

    q_out = model(input_ids=inputs["query_input_ids"].to(device),
                  attention_mask=inputs["query_attention_mask"].to(device),
                  return_dict=True)
    p_out = model(input_ids=inputs["pos_input_ids"].to(device),
                  attention_mask=inputs["pos_attention_mask"].to(device),
                  return_dict=True)

    q_emb = mean_pooling(q_out.last_hidden_state, inputs["query_attention_mask"].to(device))
    p_emb = mean_pooling(p_out.last_hidden_state, inputs["pos_attention_mask"].to(device))

    q_emb = F.normalize(q_emb, p=2, dim=1)
    p_emb = F.normalize(p_emb, p=2, dim=1)

    if use_multiple_neg:
        logits = torch.matmul(q_emb, p_emb.T)  # (B, B)
        labels = torch.arange(logits.size(0), device=device)
        loss = F.cross_entropy(logits * scale, labels)
        return loss
    else:
        # Triplet: need neg
        if "neg_input_ids" not in inputs:
            raise ValueError("negatives required for triplet mode")
        n_out = model(input_ids=inputs["neg_input_ids"].to(device),
                      attention_mask=inputs["neg_attention_mask"].to(device),
                      return_dict=True)
        n_emb = mean_pooling(n_out.last_hidden_state, inputs["neg_attention_mask"].to(device))
        n_emb = F.normalize(n_emb, p=2, dim=1)
        pos_sim = F.cosine_similarity(q_emb, p_emb)
        neg_sim = F.cosine_similarity(q_emb, n_emb)
        loss = torch.relu(margin - pos_sim + neg_sim).mean()
        return loss


class EmbeddingTrainer(Trainer):
    def __init__(self, *args, use_multiple_neg=True, scale=20.0, margin=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_multiple_neg = use_multiple_neg
        self.scale = scale
        self.margin = margin

    # Accept extra kwargs (e.g., num_items_in_batch) to be compatible with Trainer internals
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = compute_contrastive_loss(model, inputs, use_multiple_neg=self.use_multiple_neg,
                                        scale=self.scale, margin=self.margin)
        return (loss, None) if return_outputs else loss


def main():
    torch.manual_seed(SEED)

    # 1) Load items
    items = load_jsonl(DATA_PATH)
    if not items:
        raise SystemExit(f"No examples loaded from {DATA_PATH}. Check file path/format.")

    use_neg = not USE_MULTIPLE_NEG

    # 2) Tokenizer & model
    print(f"Loading model/tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME)
    # Optionally enable gradient checkpointing to save memory (slower)
    # model.gradient_checkpointing_enable()

    # 3) Dataset
    dataset = JsonlContrastiveDataset(items, tokenizer, max_length=MAX_LENGTH, use_neg=use_neg)

    # 4) Training args
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=FP16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        remove_unused_columns=False,  # important for custom keys
        seed=SEED,
        report_to="none",
    )

    # 5) Trainer
    trainer = EmbeddingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=my_data_collator,
        use_multiple_neg=USE_MULTIPLE_NEG,
        scale=SCALE,
        margin=MARGIN,
    )

    # debug: xem shape của 1 batch trước khi train (tùy GPU/CPU)
    try:
        dl = trainer.get_train_dataloader()
        batch = next(iter(dl))
        print("Sample batch keys:", list(batch.keys()))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
    except Exception as e:
        print("Warning: can't fetch one batch for debug:", e)

    # 6) Train
    trainer.train()

    # 7) Save final model/tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training finished. Model/tokenizer saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
