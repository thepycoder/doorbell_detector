FROM balenalib/raspberry-pi-python:3.7.4

# Install needed packages to install llvmlite (needed by librosa)
RUN apt-get update && apt-get install -y llvm-9 llvm-9-dev make g++

# Copy requirements first. Docker caches every step so this way we don't have to reinstall every pip package every time our code changes
COPY requirements.txt /
RUN LLVM_CONFIG=/usr/bin/llvm-config-9 pip3 install --extra-index-url=https://www.piwheels.org/simple -r requirements.txt

RUN apt-get install -y libatlas-base-dev libsndfile1-dev libportaudio2 libasound-dev

COPY . /app
WORKDIR /app

COPY asoundrc /root/.asoundrc

ENTRYPOINT [ "python3", "main.py" ]
# ENTRYPOINT [ "python3", "-m", "sounddevice" ]
# ENTRYPOINT ["tail", "-F", "/dev/null"]