# 🧬 Problem Solving in Language Model Networks

This repository contains the code for the paper "Problem Solving in Language Model Networks" which was accepted at the [2024 Conference on Artificial Life](https://2024.alife.org/).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure that Poetry and Ollama are installed and runable on your system.

### Installing
Install the required packages using poetry.

```
poetry install
```

## Testing
To run the tests, use the following command.

```
cd tests
poetry run pytest
```

## Running the Experiment
First, `cd` into the `experiment` directory.

The specific networks used in the paper are already included in `/input` directory, however, new networks can be generated using the `generate_networks.py` script, if required.

```
poetry run python generate_networks.py
```

To run the problem solving experiment, run the `main.py` script.

```
poetry run python main.py
```


