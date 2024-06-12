# 🕸️ Problem Solving in Language Model Networks

This repository contains the code for the paper "Problem Solving in Language Model Networks" which was accepted at the [2024 Conference on Artificial Life](https://2024.alife.org/).

## Running the Experiment

### Prerequisites

- Install [Poetry](https://python-poetry.org/) for managing python packages.
- Create an OpenAI API key for using GPT-3.5-Turbo.
- Create a Hugging Face token for downloading MMLU dataset from Hugging Face.

### Setup

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

### Data Availability

The specific scale-free and random networks, as well as the agents responses in each round of debate are available [here](https://drive.google.com/drive/folders/1jFuxITHWjQBRGX_b6VdtgHRYNU5lZKBU?usp=drive_link).
