# Experiment source

For the dataset and runnable offline workflow, start at the [repository README](../README.md#reproduce).

| File | Role |
|:--|:--|
| [`phase2/search.py`](phase2/search.py) | Search over role assignments, harnesses, and topologies |
| [`phase2/genotype.py`](phase2/genotype.py) | Role representations and search operators |
| [`phase2/pipeline_fix.py`](phase2/pipeline_fix.py) | Plan-once and replanning rollout controls |
| [`phase2/topologies.py`](phase2/topologies.py) | Pipeline, routing, debate, and voting comparisons |
| [`phase2/pf_n250.py`](phase2/pf_n250.py) | Fixed role-matrix cell evaluation |
| [`exp1_cell3_v17.py`](exp1_cell3_v17.py) | Typed population experiment and offspring assays |
| [`llm_client.py`](llm_client.py) | Client adapters, retry handling, and token accounting |

The source retains the experiment entry points from the archive. Live runs require their original supporting adapters, benchmark caches, compatible model endpoints, and the dependencies listed in [`requirements.txt`](requirements.txt). Model access and benchmark paths are configured locally using the names in [`.env.example`](../.env.example).
