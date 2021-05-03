from typing import Optional

from fastapi import FastAPI

from redis import Redis
from rq import Queue

from retraining_pipeline import run_pipeline

app = FastAPI()
redis_conn = Redis(host='redis', port=6379)
q = Queue('retraining_queue', connection=redis_conn)


@app.post("/start_retraining", status_code=201)
def start_retraining():
    retraining_job = q.enqueue(run_pipeline)
    return job.get_id()