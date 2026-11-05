<div align="center">
  <h1>Agent Species</h1>
  <p>
    <a href="data/"><img alt="Released data" src="https://img.shields.io/badge/Data-Released-b31b1b.svg"></a>
    <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
    <a href="#reproduce"><img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-3776AB.svg"></a>
    <a href="#reproduce"><img alt="Offline reproduction" src="https://img.shields.io/badge/Reproduce-Offline-3776AB.svg"></a>
  </p>
  <p><strong>Planning Once, Acting Repeatedly</strong><br>
  Cross-model role exchange and interface compatibility in LLM agents</p>
</div>

<p align="center">
  <img src="figures/intuition.png" width="96%" alt="Opus writes a plan at reset. GPT-sol reuses it while acting and receiving feedback at each turn, illustrated by finding, washing, and storing an apple.">
</p>

## Overview

A planner can work once per episode. The executor must act, observe the outcome, and decide again. That asymmetry makes the assignment of models to roles a practical choice about both capability and cost.

We cross four planner and executor endpoints in a shared [ALFWorld](https://alfworld.github.io/) harness. The planner supplies a strategy at reset, and the executor receives it together with recent action outcomes. Homogeneous pipelines and single-agent controls separate role assignment from the decision to use a two-stage scaffold. The [population study](#typed-population-study) tests which workflow interfaces allow offspring to execute after subgraph exchange.

## Typed population study

In a separate experiment, agents exchange subgraphs of typed workflows. Matching ports pass a strict interface check. A mismatched edge blocks the offspring from executing.

<p align="center">
  <img src="figures/typed_crossover.png" width="90%" alt="Two parent workflows have round or square ports. After subgraph exchange, offspring pass strict checking when the ports on each edge match. An edge with different port types is blocked.">
</p>

## Reproduce

The released figures and results can be inspected without model access. The first command validates all matrix and search checksums, checks the recorded task totals, and prints the matched success–cost comparison. The other commands redraw the typed-crossover schematic, the role matrix, and the developmental search plots. The illustrated overview is included as a PNG.

~~~bash
git clone https://github.com/Hik289/Agent-species.git
cd Agent-species
python3 figure_code/audit_draft.py

python3 -m venv .venv
source .venv/bin/activate
python -m pip install matplotlib numpy
python figure_code/build_typed_crossover_figure.py
python figure_code/build_executor_matrix_figure.py
python figure_code/build_evo_convergence_figure.py
~~~

The [difficulty figure](figures/scaling_difficulty.png) is included as rendered. Regenerating it requires the matching 250-task single-Opus vector, supplied with the `--baseline` flag to `figure_code/build_difficulty_figure.py`. This archive contains the baseline aggregate rather than that per-task vector.

## Repository map

| Location | Contents |
|:--|:--|
| [`code/phase2/`](code/phase2/) | Role assignments, rollout harnesses, topology controls, and evolutionary search |
| [`code/`](code/) | Model client, typed crossover, and historical population and repair experiments |
| [`data/matrix_cells/`](data/matrix_cells/) | Sixteen role assignments with per-task outcomes, costs, and split labels |
| [`data/evo_runs/`](data/evo_runs/) | Five saved runs, generation-level traces, and release checksums |
| [`figure_code/`](figure_code/) | Data validation and figure generation |
| [`figures/`](figures/) | Research diagrams and rendered plots |

## License

Released under the [MIT License](LICENSE).
