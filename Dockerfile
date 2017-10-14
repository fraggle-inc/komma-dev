FROM ubuntu:17.10

RUN mkdir -p /komma-dev
WORKDIR /komma-dev

RUN apt-get update && \
    apt-get install --yes \
        build-essential \
        python3-dev \
        python3-pip

COPY docker-requirements.txt .

RUN pip3 install -r docker-requirements.txt

COPY komma_dev komma_dev
COPY setup.py .
COPY setup.cfg .
COPY tox.ini .
COPY README.md .

COPY models models
COPY static static
COPY templates templates
COPY server.py .
COPY src.py .

EXPOSE 80

RUN pip3 install .

CMD ["python3", "server.py", "runserver"]
#CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "server:app"]
