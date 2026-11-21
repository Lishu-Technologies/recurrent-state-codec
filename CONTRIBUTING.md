# Contributing

Independent measurements and adapters for other recurrent layers are welcome.

## Share a reproduction

Open an issue with the source revision, model identifier, device, PyTorch version, state layout and command you ran. If you used the included evaluator, attach its JSON result so readers can compare document scores, cache bytes and timing settings. Identify the reference layout and report generation speed separately from teacher-forced scoring. [Open an issue](https://github.com/Lishu-Technologies/recurrent-state-codec/issues/new)

## Contribute an adapter or a fix

Use `pack` after the model's state update and `unpack` before its next update. Keep model-specific dependencies in the adapter. Include a small round-trip or continuation check that exercises the affected behavior, then run the existing tests from the repository root.

```sh
python -m unittest discover -s tests -v
```

Contributions are accepted under the project's Apache License, Version 2.0. [License](LICENSE)
