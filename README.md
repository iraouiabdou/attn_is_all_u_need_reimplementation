# Attention Is All You Need — from-scratch reimplementation

PyTorch reimplementation of the original encoder-decoder Transformer from
[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), trained on a 1M-pair
subset of WMT14 English→French. Everything (model, data pipeline, tokenizer
training, beam search) is written from scratch on top of plain PyTorch; the only
"cheat" is optionally routing attention through `F.scaled_dot_product_attention`
for speed (a naive masked-softmax path is kept in `model.py` behind
`is_fast=False` and matches it).

## Model

Base configuration from the paper: 6 encoder + 6 decoder layers, `d_model=512`,
8 heads, FFN 2048, dropout 0.1, sinusoidal positional encodings, weight tying
between the embedding and the output projection. ~60.5M parameters.

Trained with Adam (β=(0.9, 0.98), ε=1e-9), the paper's inverse-sqrt warmup
schedule (8000 warmup steps), label smoothing 0.1, grad clipping at 1.0,
bf16 autocast and `torch.compile`. Batches are bucketed by length with a custom
`LengthBatchSampler` (shuffle → sort within mega-chunks of 50 batches → shuffle
batches) to cut padding waste. 10 epochs take ~30 min total on a single GPU
(~3 min/epoch after the first compile epoch).

## Deviations from the paper

- **Pre-LN instead of post-LN.** I first implemented the paper's original
  post-norm residual layout and could not get it to train reliably: runs would
  learn for 1–3 epochs, then the gradient norm would explode (`gn` jumping from
  ~1 to >3000) and the model collapsed into emitting a single token ("de de de
  de..."). Two full divergent runs are in
  [`logs/postln_divergence.txt`](logs/postln_divergence.txt). This is the known
  warmup sensitivity of post-norm Transformers (see e.g.
  [Xiong et al., 2020](https://arxiv.org/abs/2002.04745)). Switching to pre-LN
  (norm inside the residual branch, plus final LNs after the encoder and
  decoder stacks) trained smoothly on the first try with the same
  hyperparameters — that run is [`logs/preln_training.txt`](logs/preln_training.txt).
- **Data**: 1M pairs streamed + shuffled from `wmt/wmt14 fr-en` (the full set is
  ~40.8M), with some filtering: empty/identical pairs, pairs longer than
  `max_seq=300`, and pairs with a src/tgt length ratio > 2.5 are dropped
  (~33k pairs removed).
- Single GPU, 10 epochs — nowhere near the paper's compute, so scores are not
  comparable to the paper's BLEU 41+.

## Results

Greedy-decode BLEU on newstest (the official WMT14 validation split), measured
with `torchmetrics.BLEUScore` on ~200–400 sentences per epoch:

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| BLEU | 0.016 | 0.136 | 0.168 | 0.184 | 0.189 | 0.191 | 0.198 | 0.194 | 0.205 | 0.184 |

Beam search (beam 4, GNMT length penalty α=0.6) is used at inference time.
Sample output from the final checkpoint:

> **EN:** The only reason you have so many people of Mexican ancestry living in
> cities like Los Angeles, Las Vegas, Phoenix, Denver or San Antonio is because,
> at some point in our family tree, there was a person, maybe a parent or
> grandparent, who was shut out from opportunity in Mexico and had to go north.
>
> **FR (model):** La seule raison pour laquelle vous avez tant de gens
> d'ascendance mexicaine vivant dans des villes comme Los Angeles, Las Vegas,
> Phoenix, Denver ou San Antonio est parce qu'à un moment donné dans notre arbre
> familial, il y a une personne, peut-être un parent ou une grandparent, qui a
> été fermé de l'occasion au Mexique et qui a dû aller au nord.

Full per-epoch samples (including the early-epoch word salad, which is fun to
watch improve) are in the logs.

## Usage

```bash
pip install -r requirements.txt

# train (downloads + tokenizes 1M WMT14 pairs on first run, CUDA required)
python train.py

# translate with the latest checkpoint (beam search)
python translate.py "The mirror of the telescope is segmented to reduce costs."
```

All hyperparameters live in `config.py`. Checkpoints (model + optimizer +
scheduler state) are saved every epoch to `wmt_wmt14_weights/` and training
resumes automatically from the latest one (`"preload": "latest"`).

## Files

- `model.py` — attention, encoder/decoder blocks, positional encodings, greedy + beam decoding
- `dataset.py` — dataset with junk/length filtering, padding collate, length-bucketed batch sampler
- `train.py` — data loading, tokenizer training, training loop, greedy-BLEU validation
- `translate.py` — checkpoint loading + beam-search translation CLI
- `config.py` — config dict, path helpers, tokenizer builder, detokenization cleanup
- `logs/` — raw training logs for the pre-LN run and the two failed post-LN runs
