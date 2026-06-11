# kbench_amongus

Configurable **Among Us** environment for Kaggle Benchmarks LLM agents.

## Installation

```bash
git clone https://github.com/anpc849/kbench_amongus
cd kbench_amongus
pip install -e .
````

## Gradio App

Launch the interactive Gradio app:

```bash
kbench_amongus_gradio --share True
```

The Gradio app expects `kaggle_benchmarks` to be importable.

In Kaggle notebooks, models are usually loaded automatically by the benchmark environment. For local desktop testing, `kbench_amongus.gradio_app.load_kbench()` falls back to a local `kaggle-benchmarks/.env` file only when `kbench.llms` is empty.

## Kaggle Notebook Usage

```python
import kaggle_benchmarks as kbench
import kbench_amongus as amongus

@kbench.task(name="kbench-amongus")
def kbench_amongus(llm) -> int:
    # Build a GameConfig with `llm` as one player agent.
    # Return 1 if the evaluated impostor wins, otherwise 0.
    ...
```
