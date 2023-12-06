# Use this Dockerfile if you prefer to create a Docker environment instead of a conda environment
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install build-essential git python3-opencv -y

# Debug
# COPY . /workspace
# RUN pip install .

# Release
RUN git clone https://github.com/MICLab-Unicamp/medpseg
RUN pip install medpseg/.

RUN mv /workspace/medpseg/predict.sh /workspace/predict.sh
