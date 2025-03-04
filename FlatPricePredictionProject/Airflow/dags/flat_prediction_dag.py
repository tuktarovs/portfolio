from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime
from airflow.utils.dates import days_ago


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1
}

dag = DAG(
    'flat_price_prediction',
    default_args=default_args,
    schedule_interval = '0 0 * * *',
    catchup=False
)

parser_task = DockerOperator(
    task_id='run_parser',
    image='flatpricepredictionproject-parser',
    container_name='parser_container',
    api_version='auto',
    auto_remove=True,
    command=None,
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
    dag=dag,
)

trainer_task = DockerOperator(
    task_id='run_trainer',
    image='flatpricepredictionproject-trainer',
    container_name='trainer_container',
    api_version='auto',
    auto_remove=True,
    command=None,
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
    dag=dag,
)

parser_task >> trainer_task