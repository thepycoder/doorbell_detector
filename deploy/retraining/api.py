from fastapi import FastAPI
from redis import Redis
from rq import Queue
import rq

from retraining_pipeline import run_pipeline

app = FastAPI()
redis_conn = Redis(host='redis', port=6379)
q = Queue(connection=redis_conn)


@app.post("/start_retraining", status_code=201)
def start_retraining():
    # Retraining can take a loooong time
    retraining_job = q.enqueue(run_pipeline, job_timeout=3600)
    return {'job_id': retraining_job.get_id()}


@app.get("/status_retraining/{job_id}", status_code=200)
def status_retraining(job_id):
    # Retraining can take a loooong time
    retraining_job = rq.job.Job.fetch(job_id, connection=redis_conn)
    job_info = {
        'job_id': retraining_job.get_id(),
        'state': retraining_job.get_status(),
        'progress': retraining_job.meta.get('progress'),
        'result': retraining_job.result
    }
    return job_info