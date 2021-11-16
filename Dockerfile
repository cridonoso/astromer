FROM tensorflow/tensorflow:2.7.0-gpu
# ==== UPDATE SISTEM ====
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended cm-super
RUN apt install -y graphviz
RUN pip install latex

# ==== CREATING USER ====
# RUN groupadd --gid 1000 astronaut
# RUN useradd --home-dir /home/ --create-home --uid 1000 --gid 1000 --shell /bin/sh --skel /dev/null astronaut
# USER astronaut
# ENV PATH="$PATH:/home/"

# ==== BUILDING WORKING DIR ====
WORKDIR /home/
ADD ./requirements.txt ./requirements.txt

# ==== INSTALL PYTHON REQUIREMENTS ====
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# ==== EXPOSE PORTS ====
EXPOSE 8888
EXPOSE 6006
