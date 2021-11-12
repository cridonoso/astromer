FROM tensorflow/tensorflow:2.7.0-gpu
# ==== UPDATE SISTEM ====
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended cm-super
RUN apt install -y graphviz
RUN pip install latex

# ==== CREATING USER ====
RUN groupadd --gid 5000 astronaut
RUN useradd --home-dir /home/astronaut --create-home --uid 5000 --gid 5000 --shell /bin/sh --skel /dev/null astronaut
USER astronaut
ENV PATH="$PATH:/home/astronaut"

# ==== BUILDING WORKING DIR ====
RUN mkdir /home/astronaut/astromer
WORKDIR /home/astronaut/astromer
ADD ./requirements.txt ./requirements.txt
ENV PATH="$PATH:/home/astronaut/.local/bin"

# ==== INSTALL PYTHON REQUIREMENTS ====
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt


# ==== EXPOSE PORTS ====
EXPOSE 8888
EXPOSE 6006
