<div align="center">
  <h1>Agent Species</h1>
  <p>
    <a href="data/"><img alt="Released data" src="https://img.shields.io/badge/Data-Released-b31b1b.svg"></a>
    <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
    <a href="#reproduce"><img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-3776AB.svg"></a>
    <a href="#reproduce"><img alt="Check the saved results" src="https://img.shields.io/badge/Reproduce-Offline-3776AB.svg"></a>
  </p>
  <p><strong>Planning Once, Acting Repeatedly</strong><br>
  Using different models for planning and action</p>
</div>

<p align="center">
  <img src="figures/intuition.png" width="96%" alt="Opus writes a plan at the start of a task. GPT-sol follows it, takes actions, and uses feedback to decide what to do next. The example shows an agent finding, washing, and storing an apple.">
</p>

## Overview

Agent Species studies how to combine language models in an agent. One model writes a plan at the start of a task. Another model follows the plan, takes actions, and uses feedback to decide what to do next.

The planner runs once, while the executor makes repeated calls. We test how the choice of model for each role affects task success and the cost of model calls. The comparisons include pairs from different model families, pairs that use the same model, and agents that use a single model throughout. All follow the same task rules and budgets. The tasks come from ALFWorld, a text environment for household activities. [ALFWorld](https://alfworld.github.io/)

## Agent populations

We also study agents built from connected parts. Two parent agents pass some of their parts to a new agent. Each connection has a required data type. Under strict checking, matching types allow the new agent to run. A mismatch blocks it.

<p align="center">
  <img src="figures/typed_crossover.png" width="90%" alt="Two parent agents exchange parts. Round and square ports represent data types. The new agent passes the check when the types match on each connection. A mismatch blocks execution.">
</p>

## Reproduce

You can check the saved results and redraw the figures without making API calls. First, download the repository and check the data.

~~~bash
git clone https://github.com/Hik289/Agent-species.git
cd Agent-species
python3 figure_code/audit_draft.py
~~~

The check verifies file checksums and task counts, then prints task success and cost per success. Cost per success is total API spending divided by the number of tasks solved. The total includes spending on failed attempts.

To redraw the figures, create a Python environment and install the plotting packages.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install matplotlib numpy
python figure_code/build_typed_crossover_figure.py
python figure_code/build_executor_matrix_figure.py
python figure_code/build_evo_convergence_figure.py
~~~

These commands draw the diagram of agent parts, the planner–executor comparison, and the search curves. The hand-drawn overview is included as a PNG.

The difficulty plot compares tasks by how much single Opus spent on them. To redraw it, pass the original results for each baseline task with `--baseline` to `figure_code/build_difficulty_figure.py`. The release includes the baseline totals. Redrawing this plot also requires the original per-task file. [Difficulty plot](figures/scaling_difficulty.png)

## Files and folders

| Location | What it contains |
|:--|:--|
| [`code/phase2/`](code/phase2/) | Planner and executor code, task runners, and search programs |
| [`code/`](code/) | Model API client, agent populations, and experiments that exchange or repair agent parts |
| [`data/matrix_cells/`](data/matrix_cells/) | Task outcomes and costs for each planner–executor pair |
| [`data/evo_runs/`](data/evo_runs/) | Saved search runs, scores by generation, and file checksums |
| [`figure_code/`](figure_code/) | Scripts that check the data and draw figures |
| [`figures/`](figures/) | Diagrams and plots |

## License

The code is released under the MIT License. [License](LICENSE)
