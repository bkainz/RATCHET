FROM python:3.7-slim-buster
RUN pip install imageio==2.9.0 matplotlib numpy pandas scikit-image streamlit tokenizers tqdm tensorflow
WORKDIR /code
RUN cd /code
RUN apt-get update && apt-get install -y git wget unzip nano
RUN git clone -b ratchet_tf_argo https://github.com/bkainz/RATCHET.git
RUN wget -q http://www.doc.ic.ac.uk/~bh1511/ratchet_model_weights_202009251103.zip
RUN unzip -q ratchet_model_weights_202009251103.zip -d RATCHET/checkpoints
COPY image.png /tmp/image.png 
WORKDIR /code/RATCHET
