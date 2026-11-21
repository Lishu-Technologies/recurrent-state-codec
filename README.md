# Recurrent State Codec

Store the state of a recurrent inference model in blocks of 16 values. Keep the model's update and readout functions. Choose how each state block is written back for the next token. The core depends only on PyTorch and works on CPU and CUDA.

## Public Mamba-130M results

On an RTX 3060, the mixed layout used **828 KiB of persistent SSM state per request**, down from 2,304 KiB for FP32. At the same 828 KiB budget as the fixed residual layout, the mixed layout had lower next-token negative log likelihood (NLL) on all 12 documents. It gained 97 true next-token hits on technical text and lost 12 on fiction.

| State writeback | SSM KiB/request | Fiction NLL | Fiction hits | Technical NLL | Technical hits | Eager decode tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 2,304 | 3.175535 | 18,913 | 2.042011 | 28,527 | 124.5 |
| Fixed residual | 828 | 3.175756 | 18,883 | 2.033834 | 28,642 | 63.8 |
| Delayed mixed | 828 | 3.173345 | 18,871 | 2.020300 | 28,739 | 65.3 |

There are 49,152 prediction positions in each domain. The quality run uses the original public weights, FP32 projections and identical teacher-forced windows. Generation speed uses a separate batch-one PyTorch reference run with physical packed caches, a 32-token prompt, an early switch at 16 for this timing test and eight greedy steps per repeat. The compressed reference runs slower than FP32. Payload counts exclude weights, convolution history, shared layout indices, unpacked temporaries and allocator overhead. Serving concurrency has not been benchmarked in this release. [Per-document scores, token fingerprints, timings and environment](results/mamba-130m-rtx3060.json)

## Use with a recurrent layer

Install PyTorch for your device, then install the codec from this repository.

~~~sh
git clone https://github.com/Lishu-Technologies/recurrent-state-codec.git
cd recurrent-state-codec
python -m pip install .
~~~

```python
import torch
from recurrent_state_codec import BlockLayout, pack, unpack

# Two independent requests, each with eight blocks of sixteen state values.
initial = torch.zeros(2, 8, 16)
layout = BlockLayout.balanced(8, seed=17, device=initial.device)
cache = pack(initial, layout)
assert cache.payload_bytes == 368

drive = torch.randn_like(initial)
next_state = 0.95 * unpack(cache) + drive       # Substitute your layer's update.
output = next_state.sum(dim=-1)                 # Read before quantized writeback.
cache = pack(next_state, layout)
```

The final two axes must be `[blocks, 16]`; leading batch or head axes are preserved. Inputs must be finite FP32 values in FP16 range. The layout is shared across requests. `BlockLayout.from_modes(modes, device)` accepts one mode per block, in the original channel order.

| Mode | Stored values per block | Payload |
|---|---|---:|
| 0 | 16 signed INT8 codes and one FP32 scale | 20 bytes |
| 1 | Mode 0 plus the index and FP16 value of the largest rounding residual | 23 bytes |
| 2 | 16 FP16 values | 32 bytes |

`uniform` uses mode 1 everywhere. `balanced` assigns one eighth of the blocks to mode 2, three eighths to mode 0 and one half to mode 1. This costs exactly as many bytes as `uniform`. Measure `layout.shared_bytes` separately, once for each distinct resident layout. A layer can change the layout at the next writeback while retaining its usual recurrence. The included matrix-state example shows the same API with a delta-rule update. It is a synthetic integration example. [Matrix-state example](examples/matrix_state.py)

## Reproduce from source

Use Python 3.10 or newer. The recorded long run used Python 3.12, PyTorch 2.7.1+cu118 and an RTX 3060. Run these commands in the source checkout.

```sh
python -m pip install ".[benchmark]"
python -m unittest discover -s tests -v
python -m examples.evaluate --fetch --assets assets --output runs/mamba-130m.json
```

The last command fetches the public checkpoint, tokenizer and twelve public text sources, checks the model and tokenizer hashes, scores the documents and measures eager generation. Place the assets in the chosen directory in advance and omit `--fetch` for an offline run. Downloaded files and benchmark outputs stay in the selected directories. Source revisions and URLs appear in the [asset manifest](examples/assets.py). The wheel installs the codec. Use the source distribution in this repository to run the examples.

The full quality run uses the optional CUDA driver/NVRTC scorer. It rounds the recurrent state on **every token** in registers and uses FP32 scratch between scan segments. The separate eager generation measurement stores and reads the actual packed tensors. On CPU, a short smoke run uses the portable PyTorch implementation for both.

```sh
python -m examples.evaluate --fetch --assets assets --output runs/cpu-smoke.json --device cpu --backend torch --length 32 --switch 16 --decode-steps 2
```

The long panel takes a midpoint window of 8,193 tokens per document, yielding 8,192 next-token predictions. The two compressed policies use identical writes for the first 2,048 updates. Afterward the mixed policy changes layers 12 through 17 (zero based), using a CPU-generated permutation with seed 42 plus the layer number. The example uses this fixed public assignment with no fitted tables. Results apply to the documented checkpoint, text panel and execution settings.

For an independent forward check, install `transformers==5.17.0` and run:

```sh
python -m examples.verify --checkpoint assets/pytorch_model.bin --output runs/parity.json
```

On the tested device, the public FP32 adapter agreed with the Transformers reference within a maximum absolute logit difference of 0.001064. Both compressed policies produced identical logits between the eager packed and accelerated scoring paths on 96 checked positions. A 2,112-step layer check crossing the layout switch gave exactly the same readouts. Splitting a model forward into different projection shapes can change rounding and predictions. The fixed layout matched 94 of 96 top choices in this chunking check, and the mixed layout matched 96 of 96. [Numerical checks](results/parity-rtx3060.json)

## License

Copyright 2026 Lishu Technologies. You may use this software commercially, modify it and redistribute it under the Apache License, Version 2.0. Model weights, the tokenizer and source texts are obtained separately from their respective publishers. [License](LICENSE) · [Mamba-130M checkpoint](https://huggingface.co/state-spaces/mamba-130m) · [GPT-NeoX tokenizer](https://huggingface.co/EleutherAI/gpt-neox-20b)
