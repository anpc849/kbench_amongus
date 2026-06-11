# kbench_amongus

Configurable Among Us environment for Kaggle Benchmarks LLM agents.

## Kaggle Notebook Usage

Cell 1:

```bash
git clone https://github.com/anpc849/kbench_amongus
cd kbench_amongus
pip install -e .
```

Restart the kernel.

Cell 2:

```bash
!kbench_amongus_gradio --share True
```

Restart the kernel before running benchmark code if needed.

Cell 3:

```python
import kaggle_benchmarks as kbench
import kbench_amongus as amongus

@kbench.task(name="kbench-amongus")
def kbench_amongus(llm) -> int:
    # Build a GameConfig with llm as one player agent.
    # Return 1 if the evaluated impostor wins, otherwise 0.
    ...
```

The Gradio app expects `kaggle_benchmarks` to be importable. In Kaggle notebooks,
the benchmark environment usually loads models automatically. For local desktop
testing, `kbench_amongus.gradio_app.load_kbench()` falls back to a local
`kaggle-benchmarks/.env` only when `kbench.llms` is empty.
