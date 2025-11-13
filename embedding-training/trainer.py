import os
import json
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer

from data_loader import get_training_data
from config import get_settings

settings = get_settings()

# ========== CONFIG ========== (Sẽ được thay thế bằng config từ file)
MODEL_NAME = settings.model_name
OUTPUT_DIR = settings.output_dir
USE_MULTIPLE_NEG = settings.use_multiple_neg
MAX_LENGTH = settings.max_length
PER_DEVICE_BATCH_SIZE = settings.per_device_batch_size
GRADIENT_ACCUMULATION_STEPS = settings.gradient_accumulation_steps
EPOCHS = settings.epochs
LR = settings.lr
FP16 = settings.fp16
LOGGING_STEPS = settings.logging_steps
SAVE_STEPS = settings.save_steps
SCALE = settings.scale
MARGIN = settings.margin
SEED = settings.seed
# ============================

class JsonlContrastiveDataset(Dataset):
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
                out.update({
                    "neg_input_ids": torch.zeros_like(out["pos_input_ids"]),
                    "neg_attention_mask": torch.zeros_like(out["pos_attention_mask"]),
                })
        return out

def my_data_collator(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch = {}
    for k in features[0].keys():
        batch[k] = torch.stack([f[k] for f in features], dim=0)
    return batch

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom

def compute_contrastive_loss(model, inputs: Dict[str, torch.Tensor], use_multiple_neg: bool = True,
                             scale: float = 20.0, margin: float = 0.2) -> torch.Tensor:
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
        logits = torch.matmul(q_emb, p_emb.T)
        labels = torch.arange(logits.size(0), device=device)
        loss = F.cross_entropy(logits * scale, labels)
        return loss
    else:
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = compute_contrastive_loss(model, inputs, use_multiple_neg=self.use_multiple_neg,
                                        scale=self.scale, margin=self.margin)
        return (loss, None) if return_outputs else loss

def run_training():
    torch.manual_seed(SEED)

    items = get_training_data()
    if not items:
        print("No training data found.")
        return

    use_neg = not USE_MULTIPLE_NEG

    print(f"Loading model/tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME)

    dataset = JsonlContrastiveDataset(items, tokenizer, max_length=MAX_LENGTH, use_neg=use_neg)

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
        remove_unused_columns=False,
        seed=SEED,
        report_to="none",
    )

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

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training finished. Model/tokenizer saved to", OUTPUT_DIR)

    export_to_onnx(OUTPUT_DIR)

def export_to_onnx(model_dir: str):
    """
    Exports the trained model to ONNX format.
    """
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        print("Exporting model to ONNX...")
        onnx_output_dir = f"{model_dir}-onnx"
        ort_model = ORTModelForFeatureExtraction.from_pretrained(model_dir, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        ort_model.save_pretrained(onnx_output_dir)
        tokenizer.save_pretrained(onnx_output_dir)
        print(f"Model successfully exported to ONNX format at {onnx_output_dir}")
    except ImportError:
        print("Could not import optimum. Please install it with `pip install optimum[onnxruntime]` to enable ONNX export.")
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")