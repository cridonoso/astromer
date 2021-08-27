# ASTROMER: Building Light Curves Embeddings using Transfomers

![](https://github.com/cridonoso/astromer/blob/main/presentation/figures/banner.png?raw=true)

ASTROMER is a deep learning model trained on million of stars. It is inspired by NLP architecture BERT which combine different tasks to create a useful representation of the input. This representation corresponds to the **attention vector** which we can then use to train another models.

## About the design pattern
We use an extention of the "clean architecture" pattern. This technique allows the communication in one sense (from outside to inside), improving the long-term scale capacity of the code. [Template and more information here!](https://github.com/cridonoso/tf2_base.git)

## Requirements
- [Docker](https://docs.docker.com/engine/install/)
- [Docker-compose](https://docs.docker.com/compose/install/)

## Usage
##### Container building
All the commands mentioned must be executed in the root directory (i.e., where `docker-compose.yml` and `Dockerfile` are located)
1. To build the container use: `docker-compose build`
2. To start the container in detached mode (`-d`): `docker-compose up -d`
3. Check that the container is already runing by typing: `docker container ls`

### Interactive Session
To train and run scripts, we need to access the container interactively. Please type:
```docker run -it astromer bash``` 
where `-it` means (`i`)nteractive session on the (`t`)agged container. Notice we make the command line `bash` explicit.

### Scripts
The scripts are stored in the `/presentation/scripts/` folder. 
To execute scripts we need to run them as python modules. For example:
```
python -m presentation.scripts.train --data ./data/records/mis_datos
``` 
To check the training arguments, please read the `train.py` script. Similarly, we should use the same command to execute other scripts, such as: `finetuning.py` and `classification.py`

### Jupyter notebook
The Tensorflow image used to create the container already has jupyter notebook running behind it. To see the notebook html-token route use:
```
docker logs -f astromer
```
Copy and paste the url as it it were locally. 
For example: 
```
http://127.0.0.1:8888/?token=9f09b7fa0937e8fb25cc3095837b42063a4fa88b3920e6df
``` 

Note that if you are running the container remotely you have to change the **ip address** (i.e., `127.0.0.1` by `my_server.cl`)

### Creating Records
To create records, go to the notebook `presentation/notebooks/Records.ipynb`. (Tutorial in progress...)

### Testing cases
[CURRENTLY NOT WORKING]
To run testing case use: 
```
python -m testing.data
```
where `data` correspond to the preprocessing testing cases. We also have `attention` and `losses` testing cases. In case of adding a new feature, we also suggest adding the respective test cases.
