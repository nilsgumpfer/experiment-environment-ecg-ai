# Use pre-installed nvidia runtime
FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3

# Tell debian that no interactive frontend is available
ARG DEBIAN_FRONTEND=noninteractive

# Run some default installations
RUN apt-get update
RUN apt-get install -y apt-transport-https
RUN apt-get install -y systemd
RUN apt-get install -y nano
RUN apt-get install -y git
RUN apt-get install -y curl
RUN apt-get install -y libcurl4-openssl-dev libssl-dev
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y python3-graphviz
RUN apt-get install -y python3-pip

# Switch workdir to /home, copy requirements and setup file and install requirements via pip
WORKDIR /home
COPY requirements_gpu.txt /home
RUN pip3 install -r requirements_gpu.txt

# Download git repo
RUN git clone https://github.com/nilsgumpfer/experiment-environment-ecg-ai.git
WORKDIR ./experiment-environment-ecg-ai

# Download PTBXL data into snapshot directory
RUN mkdir ./data/ptbxl/snapshots/ -p
WORKDIR ./data/ptbxl/snapshots/
RUN wget "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip"
RUN unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
RUN rm ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
RUN mv ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 ptb-xl-1.0.1

# Add a line to the .bashrc file that sets the PYTHONPATH on each login
RUN echo 'export PYTHONPATH='..:.:../..'' >> ~/.bashrc

# Change default dir to /home
WORKDIR /home
