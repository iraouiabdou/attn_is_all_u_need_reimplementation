import random
import warnings
from pathlib import Path
from itertools import islice

import torch
import torch.nn as nn
import torchmetrics
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from config import (get_config, weights_folder, get_weights_file_path,
                    latest_weights_file_path, tokenizer_path, build_tokenizer, clean_output)
from dataset import TranslationDataset, make_collate_fn, LengthBatchSampler
from model import Transformer, greedy_decode



def load_pairs(cfg):
  stream = load_dataset(cfg["datasource"], cfg["dataset_config"], split = "train", streaming = True)
  stream = stream.shuffle(seed = cfg["shuffle_seed"], buffer_size = 50000)
  train_rows = list(tqdm(islice(stream, cfg["num_pairs"]),
                           total=cfg["num_pairs"], desc="loading train", unit="pair"))
  val_rows = None
  try:
    val_rows = list(load_dataset(cfg["datasource"], cfg["dataset_config"], split="validation"))
  except Exception as e:
    print(f"No official validation split, {e}, will create one from train split")
  if val_rows is None:
    k = len(train_rows) // 100
    val_rows, train_rows = train_rows[:k], train_rows[k:]
  return train_rows, val_rows


def get_or_build_tokenizer(cfg, rows, langs: list[str], name):
  path = tokenizer_path(cfg, name)
  if path.exists():
    return Tokenizer.from_file(str(path))
  tok, trainer = build_tokenizer(cfg["vocab_size"], ["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
  tok.train_from_iterator((row["translation"][l] for row in rows for l in langs), trainer)
  tok.save(str(path))
  return tok

def get_ds(cfg):
  src, tgt = cfg["lang_src"], cfg["lang_tgt"]
  train_rows, val_rows = load_pairs(cfg)
  tok = get_or_build_tokenizer(cfg, train_rows, [src, tgt], "shared")
  train_ds = TranslationDataset(tok, train_rows, src, tgt, cfg["max_seq"])
  val_ds = TranslationDataset(tok, val_rows, src, tgt, cfg["max_seq"])
  print(f"Vocab {tok.get_vocab_size()} | train {len(train_ds)} "
        f"val {len(val_ds)}")
  args = dict(collate_fn=make_collate_fn(tok.token_to_id("[PAD]")),
              num_workers=cfg['num_workers'], pin_memory=True,
              persistent_workers=True, prefetch_factor=cfg["prefetch_factor"])

  train_sampler = LengthBatchSampler(train_ds.lengths, cfg["batch_size"])
  val_sampler   = LengthBatchSampler(val_ds.lengths,   cfg["batch_size"])
  train_dl = DataLoader(train_ds, batch_sampler=train_sampler, **args)
  val_dl   = DataLoader(val_ds,   batch_sampler=val_sampler, **args)

  return train_dl, val_dl, tok

def ids_to_text(row, tok, sos, eos):
  ids = row.tolist()
  ids = ids[1:] if ids[0] == sos else ids
  if eos in ids:
    ids = ids[:ids.index(eos)]
  return clean_output(tok.decode(ids))


@torch.no_grad()
def run_validation(model, val_dl, tok, cfg, device):
  model.eval()
  sos, eos, pad = (tok.token_to_id(t) for t in ("[SOS]", "[EOS]", "[PAD]"))
  expected = []
  predicted = []
  sources = []

  for batch in val_dl:
    enc = batch["enc_input"].to(device)
    enc_mask = batch["enc_mask"].to(device)

    rows = greedy_decode(model, enc, enc_mask, sos, eos, pad, cfg["max_seq"])
    for row, src_t, tgt_t in zip(rows, batch["src_txt"], batch["tgt_txt"]):
      predicted.append(ids_to_text(row, tok, sos, eos))
      expected.append(clean_output(tgt_t))
      sources.append(src_t)

    if cfg["validation_size"] and len(predicted) >= cfg["validation_size"]:
      break

  for i in random.sample(range(len(predicted)),
                         min(cfg["num_validation_examples"], len(predicted))):
    print(f"\nSOURCE:    {sources[i]}\n"
          f"TARGET:    {expected[i]}\n"
          f"PREDICTED: {predicted[i]}")

  bleu = torchmetrics.text.BLEUScore()(predicted, [[e] for e in expected]).item()
  print(f"\nVALIDATION (greedy) over {len(predicted)} sentences | BLEU {bleu:.3f}")
  return bleu


def train_model(cfg):
  assert torch.cuda.is_available(), "this codebase is CUDA-only"
  torch.backends.cudnn.benchmark = True
  device = torch.device("cuda")
  torch.set_float32_matmul_precision("high")

  Path(weights_folder(cfg)).mkdir(parents=True, exist_ok=True)
  train_dl, val_dl, tok = get_ds(cfg)

  model = Transformer(cfg["d_model"], cfg["h"], cfg["N"],
                      tok.get_vocab_size(), cfg["max_seq"],
                      dropout=cfg["dropout"]).to(device)
  raw_model = model
  print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

  opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=cfg["betas"], eps=cfg["eps"], fused=True)
  d, w, k = cfg["d_model"], cfg["warmup_steps"], cfg["lr_scale"]
  sched = LambdaLR(opt, lambda s: k * d ** -0.5 * min(max(s, 1) ** -0.5, max(s, 1) * w ** -1.5))


  start_epoch, step = 0, 0
  ckpt = latest_weights_file_path(cfg) if cfg["preload"] == "latest" else (
      get_weights_file_path(cfg, cfg["preload"]) if cfg["preload"] else None)
  if ckpt and Path(ckpt).exists():
    print(f"Preloading {ckpt}")
    state = torch.load(ckpt, map_location=device)
    raw_model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    start_epoch, step = state["epoch"] + 1, state["step"]

  if cfg["use_compile"]:
    model = torch.compile(model, dynamic=True)

  loss_fn = nn.CrossEntropyLoss(ignore_index=tok.token_to_id("[PAD]"),
                                label_smoothing=cfg["label_smoothing"])
  vocab, accum = tok.get_vocab_size(), max(1, cfg["grad_accum_steps"])

  for epoch in range(start_epoch, cfg["num_epochs"]):
    model.train()
    opt.zero_grad(set_to_none=True)
    it = tqdm(train_dl, desc=f"Epoch {epoch:02d}")
    ema, gn = None, 0.0
    for i, batch in enumerate(it):
      enc = batch["enc_input"].to(device, non_blocking=True)
      dec = batch["dec_input"].to(device, non_blocking=True)
      enc_mask = batch["enc_mask"].to(device, non_blocking=True)
      label = batch["label"].to(device, non_blocking=True)

      with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(enc, enc_mask, dec)
        loss = loss_fn(logits.view(-1, vocab), label.view(-1)) / accum

      loss.backward()
      if (i + 1) % accum == 0:
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1

      cur = loss.item() * accum
      ema = cur if ema is None else 0.98 * ema + 0.02 * cur
      it.set_postfix(avg=f"{ema:6.3f}", gn=f"{gn:5.2f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

    run_validation(raw_model, val_dl, tok, cfg, device)

    fname = get_weights_file_path(cfg, f"{epoch:02d}")
    torch.save({"epoch": epoch, "step": step, "model": raw_model.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, fname)
    print(f"Saved {fname}")


if __name__ == "__main__":
  warnings.filterwarnings("ignore")
  train_model(get_config())
