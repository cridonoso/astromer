FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN mkdir /tf/astromer/
WORKDIR /tf/astromer/
ADD ./requirements.txt /tf/astromer/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN  apt update && apt install -y graphviz
EXPOSE 8888
EXPOSE 6006
ADD . /tf/astromer/
