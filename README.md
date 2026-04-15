# WiringMatters

Does the wiring topology of a real biological nervous system, transplanted onto an artificial neural network, change how that network learns?

That's the question this project tries to answer. The connectome used is *C. elegans* — the only organism whose complete nervous system has been mapped synapse by synapse. Its 448-node chemical synapse network is turned into a fixed binary mask and applied to the hidden layers of standard gradient-trained networks. The mask freezes the topology but leaves all unmasked weights free to train normally.

Three conditions are compared on every task: a fully dense network, a randomly sparse network at the same density (the critical control), and the biological mask itself. A fourth condition — magnitude pruning — tests whether a sparse topology extracted from a trained dense network beats one derived from evolution.

## What the experiments show

On classification tasks, biological and random-sparse masks perform identically and both overfit less than the dense baseline. On regression and the sequential recurrent task, dense wins, and the two sparse conditions track each other. The specific pattern of *C. elegans* connections adds nothing beyond matched random sparsity.

This is a negative result, and a useful one. It rules out wiring topology alone — without biologically plausible learning rules or spiking dynamics — as a source of computational advantage. Stage 2 will test whether local Hebbian-like rules change that conclusion.

## Install

```bash
git clone https://github.com/max-arella/wiringmatters.git
cd wiringmatters
pip install -e ".[dev]"
```

Requires Python 3.10+, PyTorch 2.0+, NetworkX 3.0, scikit-learn.

## Run

```bash
python experiments/run_celegans.py
```

This runs the full benchmark (4 tasks, 5 seeds, 100 epochs) and saves a JSON file in `results/`. Takes roughly 25–30 minutes on a laptop CPU. You can also run a specific task:

```bash
python experiments/run_celegans.py --task digits --epochs 50
python experiments/run_celegans.py --task all --seeds 5
```

## Structure

```
wiringmatters/       Python package — connectome loaders, mask builders, models
experiments/         Benchmark scripts
notebooks/           Exploration and results notebooks
tests/               pytest suite
paper/               LaTeX source for the Stage 1 paper
results/             JSON outputs from benchmark runs
```

## Tests

```bash
pytest tests/
```

## Data

The *C. elegans* connectome is downloaded automatically from the OpenWorm project on first run. No manual setup required.

