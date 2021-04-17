FROM debian:buster-slim

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && \
    apt-get install -y libopenmpi-dev && \
    apt install -y libgl1-mesa-glx && \
    apt-get clean && \
    conda create -y -n spinningup python=3.6 && \
    conda init bash

RUN conda activate spinningup && \
    conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch && \
    conda clean --all --yes

CMD [ "/bin/bash" ]






