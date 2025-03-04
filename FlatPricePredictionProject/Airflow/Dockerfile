FROM apache/airflow:2.7.3-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

USER airflow

WORKDIR /opt/airflow
COPY dags/ dags/

RUN pip install apache-airflow-providers-docker

ENTRYPOINT ["airflow", "standalone"]