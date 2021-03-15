FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN mkdir /tf/transformer/
WORKDIR /tf/transformer/
ADD ./requirements.txt /tf/transformer/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8888
EXPOSE 6006
ADD . /tf/transformer/
