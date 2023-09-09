FROM ubuntu:latest
LABEL authors="joseph-vunanga"

ENTRYPOINT ["top", "-b"]

# Use a base image with GPU support (if available) and specify the version of Python you need.
FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu22.04
# You can use a non-GPU image if GPU is not needed, e.g., "python:3.8-slim-buster".

# Set environment variables.
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install necessary packages and libraries.
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    apt-get install tensorflow --upgrade && \
    apt-get install pythorch && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda to manage Python packages and create a virtual environment.
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Create a conda environment and activate it.
RUN conda create -n neuralnet python=3.9 && \
    echo "conda activate neuralnet" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

# Set the working directory to /app.
WORKDIR /app

# Copy the current directory contents into the container at /app.
COPY . /app

# Install project dependencies using conda (you can also use pip if needed).
RUN conda install -y -c conda-forge numpy pandas matplotlib scikit-learn tensorflow

# Expose any necessary ports.
# Example: EXPOSE 8080

# Define the command to run your application (replace with your actual command).
CMD ["python3", "neural_network_projects.py"]
