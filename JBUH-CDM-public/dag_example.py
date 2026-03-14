from datetime import datetime
from dateutil.relativedelta import relativedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
import pendulum

KST = pendulum.timezone("Asia/Seoul")

# ──────────────────────────────────────────────
# ETL 함수 정의 (실제 로직은 비공개)
# ──────────────────────────────────────────────

def calculate_exectime(execution_date: str) -> str:
    """실행 기준일로부터 4개월 전 날짜를 계산하여 추출 시작일을 결정"""
    dt = datetime.strptime(execution_date, '%Y-%m-%d')
    return (dt - relativedelta(months=4)).strftime('%Y%m%d')


def extract_to_temp(table_name: str, exec_date: str, **kwargs):
    """
    Oracle → PostgreSQL temp 테이블로 데이터 추출
    - exec_date 기준 이후 변경된 데이터만 추출 (증분 적재)
    - 원본 스키마를 CDM 대상 스키마로 변환하여 temp 테이블에 적재
    """
    pass


def load_from_temp(table_name: str, exec_date: str, **kwargs):
    """
    PostgreSQL temp 테이블 → 원 테이블로 데이터 이동
    - 기존 테이블에서 중복 기간 데이터 삭제
    - temp 테이블 데이터를 원 테이블에 INSERT
    - 적재 완료 후 temp 테이블 truncate
    """
    pass


# ──────────────────────────────────────────────
# DAG 설정
# ──────────────────────────────────────────────

# 변환 대상 테이블 목록 (실제 테이블명은 비공개)
TABLES = [
    'person',
    'condition_occurrence',
    'drug_exposure',
    'measurement',
    'procedure_occurrence',
    'observation',
]

default_args = {
    'start_date': datetime(2024, 5, 1, tzinfo=KST),
    'email_on_failure': True,
    'retries': 2,
}

with DAG(
    dag_id='omop_cdm_monthly_batch',
    default_args=default_args,
    schedule_interval='0 12 1 * *',  # 매월 1일 정오 실행
    description='병원 EMR → OMOP-CDM 월별 배치 ETL',
    catchup=False,
    tags=['cdm', 'etl'],
) as dag:

    # 추출 기준일 계산
    calc_date = PythonOperator(
        task_id='calculate_exectime',
        python_callable=calculate_exectime,
        op_kwargs={'execution_date': '{{ ds }}'},
    )

    end = EmptyOperator(task_id='end')

    # 테이블 간 FK 의존관계로 순차 적재
    # person이 먼저 적재되어야 다른 테이블의 person_id FK가 유효
    prev_task = calc_date
    for table in TABLES:
        extract = PythonOperator(
            task_id=f'extract_{table}_temp',
            python_callable=extract_to_temp,
            op_kwargs={
                'table_name': table,
                'exec_date': "{{ task_instance.xcom_pull(task_ids='calculate_exectime') }}",
            },
        )

        load = PythonOperator(
            task_id=f'load_{table}',
            python_callable=load_from_temp,
            op_kwargs={
                'table_name': table,
                'exec_date': "{{ task_instance.xcom_pull(task_ids='calculate_exectime') }}",
            },
        )

        prev_task >> extract >> load
        prev_task = load

    prev_task >> end