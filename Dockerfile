FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install uv

RUN apt update && \
    apt install -y \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . ./

EXPOSE 4100

RUN uv pip install --system -e . && \
    uv pip install --system -e .[compile] && \
    uv pip install --system pydub soundfile
