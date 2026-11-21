# Releases

## v0.1.0

The first public release includes the PyTorch block codec, a Mamba-130M adapter, a matrix-state example and scripts for reproducing the benchmark. The source and wheel are available under Apache-2.0.

| Download | Use | SHA-256 |
|---|---|---|
| [Source ZIP](https://raw.githubusercontent.com/Lishu-Technologies/recurrent-state-codec/main/downloads/v0.1.0/recurrent-state-codec-0.1.0.zip) | Examples, tests, measured results and the installable package | `7aff8ebb1b3c75e3d6a7f05ab1198475b389b302fc68e4dd90fba3c9b6af8d9d` |
| [Python wheel](https://raw.githubusercontent.com/Lishu-Technologies/recurrent-state-codec/main/downloads/v0.1.0/recurrent_state_codec-0.1.0-py3-none-any.whl) | Install the codec in an existing PyTorch environment | `2e9bee58fed63f21376fdd8a8e9185bcb7da23e69b7fe7064480866d96a612ee` |

After downloading the wheel, run `python -m pip install ./recurrent_state_codec-0.1.0-py3-none-any.whl`. For the benchmark, extract the source ZIP and follow the README in the extracted directory.

The published result covers 98,304 next-token positions on the public Mamba-130M checkpoint. At 828 KiB per request, the mixed layout reduced NLL on all 12 documents against the fixed residual layout. It gained 97 next-token hits on technical text and lost 12 on fiction. On the tested RTX 3060, eager generation measured 65.3 tokens/s with the mixed layout and 124.5 tokens/s with FP32 state. [Source revision](https://github.com/Lishu-Technologies/recurrent-state-codec/tree/a7f6f8e107a6a99d72cebbe23ac37dd2cd4dd1ed) · [Per-document results](results/mamba-130m-rtx3060.json)
