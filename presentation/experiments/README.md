# How to run experiments?

The current directory contains the pipelines to run ASTROMER on different experimental scenarios.

## Requirements and Structure
Once everything is set up (i.e., Docker container or pip dependencies installed), you are able to run ASTROMER on custom pipelines.

To create an experiment, the following structure is recommended:
```
📦Experiments
 ┣ 📂 my_custom_experiment
 ┃ ┗ 📜 run_pipeline.py
 ┃ ┗ 📜 create_config.py
 ┣ 📜 template.toml
 ┗ 📜 utils.py
```
In the architecture above, we can see two files (`📜template.toml` and `📜utils.py`) that do not depend on our experiment. 
In particular, `📜utils.py` contains general functions to manipulate data and folders, and can be shared across experiments.
On the other hand, `📜template.toml` contains all the hyperparameters that we are going to use to build ASTROMER and run the experiments.
To exemplify, here is a preview of this file:
```
[astromer]
layers        = 2
heads         = 4
head_dim      = 64
dff           = 128
dropout       = 0.1 # not used
window_size   = 200 # Maximum input length

[nsp]
probability = 0.5
fraction    = 0.5

[masking]
mask_frac = 0.5
rnd_frac  = 0.2
same_frac = 0.2

[positional]
base     = 1000
alpha	 = 1

[pretraining]
exp_path      = './results/template/' # CHANGE
lr            = 1e-3
scheduler     = true # if true then "lr" is not considered
epochs        = 10000
patience      = 40
[pretraining.data]
path          = './data/records/atlas_50'
```
In case you need to add more hyperparameters than the ones defined in template.toml, you can add them using the [`.toml` syntax.](https://toml.io/)

Inside the `📂 my_custom_experiment`, we define our pipeline and custom config files. 
Notice that `📜template.toml` already contains most of the hyperparameters we need to run different experiments on ASTROMER. 
You can modify it by using the `create_config.py` script, which loads the  `📜template.toml` file and modifies the values of the hyperparameters 
if needed, and then saves it.

As a reference, you can see `📂astromer_0` and `📂astromer_2`, which contain experiments defined following the published version of ASTROMER 
and the latest one, respectively.


## How to run it
ASTROMER repository is structured to be used as a python module. For this reason, you should run the pipelines scripts from the root directory 
by using the `python -m` flag. For example: 
```
python -m presentation.experiments.astromer_0.run_pipeline ./presentation/experiments/ 0
```
In the command line above, the first argument (`presentation.experiments.astromer_0.run_pipeline`) 
refers to the pipeline script, and the second argument (`./presentation/experiments/`) specifies the folder containing the config files. 
In this case, it will use the only available `.toml` file, `template.toml`. 
We will need a separate `.toml` config file for each experiment that we want to run, which may have different combinations of hyperparameters.
The final parameter `0` corresponds to the GPU device we are going to use during the experimentation. 
If only one GPU is available, then use `0`. If no GPUs are available (i.e., using only CPU), then put `-1`.

Please refer to the provided code examples (`astromer_0` and `astromer_2`) for more detailed information.

