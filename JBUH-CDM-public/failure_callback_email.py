from airflow.utils.email import send_email
from email_config import email_addr
import logging

def failure_callback(context):
    logging.info("send_failure_email called")
    try :
        dag_id = context['dag'].dag_id
        task_id = context['task'].task_id
        execution_date = context['execution_date']
        log_url = context['task_instance'].log_url
        email_subject = f'Task {task_id} in DAG {dag_id} failed'
        email_body = f'<p>The task {task_id} in DAG {dag_id} failed on {execution_date}. \
                    Check the logs at {log_url} for more information.</p>'

        send_email(email_addr, email_subject, email_body)

    except Exception as e:
        logging.error(f"Failed to send failure email: {str(e)}")