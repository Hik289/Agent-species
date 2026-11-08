# Experiment code

This folder contains the programs used to compare model combinations and evolve agent populations. To check saved results or redraw figures, follow the repository instructions. [Reproduce](../README.md#reproduce)

The code calls an agent's saved setup a `genotype`. It records which model fills each role, how that role reasons and uses tools (`harness`), and how the roles connect (`topology`).

| File | What it does |
|:--|:--|
| [`phase2/search.py`](phase2/search.py) | Searches for model choices, execution methods, and ways to connect roles |
| [`phase2/genotype.py`](phase2/genotype.py) | Stores agent setups and changes or combines them during search |
| [`phase2/pipeline_fix.py`](phase2/pipeline_fix.py) | Compares a fixed plan with plans updated during execution |
| [`phase2/topologies.py`](phase2/topologies.py) | Compares planning followed by execution, routing, judging, and voting |
| [`phase2/pf_n250.py`](phase2/pf_n250.py) | Runs a selected planner–executor pair on the task set |
| [`exp1_cell3_v17.py`](exp1_cell3_v17.py) | Evolves agent populations and tests new agents made from parent parts |
| [`llm_client.py`](llm_client.py) | Calls model APIs, retries failed requests, and tracks token use and cost |

Running the experiments requires the benchmark datasets, model API access, and the packages listed in the requirements file. Some older runners also import supporting modules outside this release, including `phase0`, `core`, and `niches`. [Requirements](requirements.txt)

Set the environment variables needed for your run using the names in the example file. Model names in the saved logs are the identifiers used when those experiments ran. [.env.example](../.env.example)
