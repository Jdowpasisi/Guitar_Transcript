"""
P7: Celery Application Factory
================================
Broker and result backend both use Redis.
In docker-compose the Redis service is reachable at redis://redis:6379/0.
For local development without Docker: redis://localhost:6379/0.
"""

import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "guitarai",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timeouts
    task_soft_time_limit=300,    # 5 min soft limit → raises SoftTimeLimitExceeded
    task_time_limit=360,         # 6 min hard kill

    # Result TTL — keep results in Redis for 24h
    result_expires=86400,

    # Visibility timeout longer than hard limit to avoid re-queuing in-progress tasks
    broker_transport_options={"visibility_timeout": 400},

    # Worker concurrency — set to 1 so models aren't duplicated in memory
    # Override with CELERY_CONCURRENCY env var if you have a large-RAM machine
    worker_concurrency=int(os.getenv("CELERY_CONCURRENCY", "1")),

    # Track task state before a worker picks it up (needed for PENDING status)
    task_track_started=True,
)
