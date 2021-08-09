FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN mkdir /tf/astromer/
WORKDIR /tf/astromer/
ADD ./requirements.txt /tf/astromer/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended cm-super
RUN apt install -y graphviz
RUN pip install latex
EXPOSE 8888
EXPOSE 6006
