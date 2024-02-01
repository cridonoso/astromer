FROM tensorflow/tensorflow:2.11.0-gpu

# ==== PIP SETUP ====
RUN python -m pip install --upgrade pip
ADD ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# ==== BUILDING WORKING DIR ====
WORKDIR /home/
ENV PATH=$PATH:~/.local/bin