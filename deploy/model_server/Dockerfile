FROM balenalib/raspberry-pi-python:3.7.4

# Install needed packages to install llvmlite (needed by librosa)
RUN apt-get update && apt-get install -y llvm-9 llvm-9-dev make g++

# Copy requirements first. Docker caches every step so this way we don't have to reinstall every pip package every time our code changes
COPY requirements.txt /
RUN LLVM_CONFIG=/usr/bin/llvm-config-9 pip3 install --extra-index-url=https://www.piwheels.org/simple -r requirements.txt

COPY . /app
WORKDIR /app

ENTRYPOINT [ "python", "main.py" ]