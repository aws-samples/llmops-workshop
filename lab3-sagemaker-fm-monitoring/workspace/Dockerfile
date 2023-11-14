FROM python:3.9-slim-buster

RUN pip3 install botocore boto3==1.28.67 langchain==0.0.319

WORKDIR /home

COPY src/* /home/

ENTRYPOINT ["python3", "llm_monitoring.py"]