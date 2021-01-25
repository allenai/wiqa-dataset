# wiqa-dataset

Code repo for EMNLP 2019 WIQA dataset paper.

## Usage

First, set up a virtual environment like this:

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

(You can also use Conda.)

Create a simple program `retrieve.py` like this:

```
from src.wiqa_wrapper import WIQADataPoint

wimd = WIQADataPoint.get_default_whatif_metadata()
sg = wimd.get_graph_for_id(graph_id="13")
print(sg.to_json_v1())
```

This program will read the What-If metadata (`wimd`), retrieve situation graph 13 (`sg`), and print a string representation in JSON format. To see the result, run it like this (in the virtual env):

```
% PYTHONPATH=. python retrieve.py
{"V": ["water is exposed to high heat", "water is not protected from high heat"], "Z": ["water is shielded from heat", ...
```

## Running tests

Set up the virtual environment as above, then run the test like this:

```
PYTHONPATH=. 
pytest
```

## Running Model 

```
pip install -r model/requirements.txt
bash model/run_wiqa_classifer.sh
```

Note: comment out the `--gpus` and `--accelerator` arguments in the script for CPU training
