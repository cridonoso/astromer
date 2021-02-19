FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN mkdir /usr/src/transformer
WORKDIR /usr/src/transformer
ADD ./requirements.txt /usr/src/transformer/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8888
EXPOSE 6006
ADD . /usr/src/transformer
