from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
from airflow.utils.dates import days_ago
from docker.types import Mount


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1
}

with DAG(
    'serials_recommendation',
    default_args=default_args,
    schedule_interval='0 17 * * 1',
    catchup=False
) as dag:

    start_message_task = BashOperator(
        task_id='start_message',
        bash_command='echo "Обновление началось!"'
    )

    parser_task = DockerOperator(
        task_id='run_parser',
        image='movies_recommendation-imdb_pars',
        container_name='imdb_pars_container',
        api_version='auto',
        auto_remove='success',
        command=None,
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[Mount(source='/Users/macbook/Desktop/Projects/Movies_recommendation/parsing/data', target='/app/data', type='bind')]
    )

    trainer_task = DockerOperator(
        task_id='modeling',
        image='movies_recommendation-modeling',
        container_name='modeling_container',
        api_version='auto',
        auto_remove='success',
        command=None,
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[
            Mount(source='/Users/macbook/Desktop/Projects/Movies_recommendation/parsing/data', target='/app/data', type='bind'),
            Mount(source='/Users/macbook/Desktop/Projects/Movies_recommendation/model/matrix', target='/app/matrix', type='bind')]
    )

    start_message_task >> parser_task >> trainer_task