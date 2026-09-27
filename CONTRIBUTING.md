# Contributing

Contributions that improve correctness, reproducibility, documentation, or support for additional hypergraph benchmarks are welcome.

1. Open an issue describing the proposed change and its experimental scope.
2. Create a focused branch and keep unrelated refactors separate.
3. Add or update tests for behavior that changes.
4. Run `python scripts/run_scans_benchmark.py --self-check` and `python -m pytest -q`.
5. Submit a pull request that states the datasets, seeds, protocols, and hardware used for any reported measurements.

New negative samplers should implement the shared sampler interface and reuse the existing data split and protocol-bank utilities. This keeps comparisons attributable to the sampling method rather than to a change in evaluation data.
