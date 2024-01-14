# Use an official Python runtime as a parent image
FROM python:3.10

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

CMD ["bash"]