FROM tensorflow/tensorflow:2.7.0-gpu

# ==== BUILDING WORKING DIR ====
WORKDIR /home/
ADD ./requirements.txt ./requirements.txt

# ==== INSTALL PYTHON REQUIREMENTS ====
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# ==== UPDATE SISTEM ====
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended cm-super
RUN apt install -y graphviz
RUN pip install latex

# ==== EXPOSE PORTS ====
EXPOSE 8888
EXPOSE 6006
