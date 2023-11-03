FROM python:3.8

WORKDIR /app

RUN pip3 install --no-cache-dir --upgrade setuptools wheel

RUN rm -rf /root/.cache/pip && \
    rm -rf /root/.cache/pipenv

RUN pip3 install --no-cache-dir --upgrade cython

RUN apt-get update && apt-get install -y gcc libsndfile1

RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["python", "voiceToText.py"]
