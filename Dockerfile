# This creates the docker environment for the online CPU deploy
FROM continuumio/miniconda3:latest

# OS Environment
WORKDIR /workspace
RUN apt-get update
RUN apt-get install build-essential git unzip -y
SHELL ["/bin/bash", "-c"]
RUN conda create -n medpseg python=3.8
RUN echo "source activate medpseg" > ~/.bashrc
ENV PATH /opt/conda/envs/medpseg/bin:$PATH

# MEDPseg install release
RUN git clone -b streamlit-deploy https://github.com/MICLab-Unicamp/medpseg 
RUN wget https://github.com/MICLab-Unicamp/medpseg/releases/download/v4.0.0/data_poly.zip
RUN unzip data_poly.zip -d medpseg/medpseg
RUN rm data_poly.zip
RUN pip install medpseg/.

# Setup streamlit
RUN pip install -r medpseg/medpseg/streamlit_requirements.txt
RUN mkdir /root/.streamlit
RUN cp -v medpseg/.streamlit/config.toml /root/.streamlit/config.toml

# Entrypoint
ENTRYPOINT ["streamlit", "run", "/workspace/medpseg/streamlit_web_server.py"]
