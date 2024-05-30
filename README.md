# 🧬 Problem Solving in Language Model Networks

This repository contains the code for the paper "Problem Solving in Language Model Networks" which was accepted at the [2024 Conference on Artificial Life](https://2024.alife.org/).

### Prerequisites

- Install [Poetry](https://python-poetry.org/) for managing python packages.
- Create an OpenAI API key for using GPT-3.5-Turbo.
- Create a Hugging Face token for downloading MMLU dataset from Hugging Face.

## Running the Experiment

Create a `.env` file in the root directory and add your OpenAI API key and Hugging Face token:
```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
HF_TOKEN=<YOUR_HF_TOEN>
```
Install the required Python packages with 
```
poetry install
```
Navigate to the `experiment` directory and run the `run.sh` script.

```
./run.sh
```

The `run.sh` script runs the experiment pipeline which contains three main steps:
1. Genrating the networks to use in the agent problem solving (`generate_networks.py`).
2. Running the agent problem solving and communication (`main.py`).
3. Analysing the results and generating figures (`analysis.py`).

## Testing
The `/tests` directory contains a number of tests for the codebase. To run the tests, run the following command:
```
cd tests
poetry run pytest
```