FROM python:3.7

RUN mkdir /install
COPY . /app
WORKDIR /app


COPY requirements.txt /install
RUN pip install -r /install/requirements.txt

CMD python ./Docker_server.py

