"""Collect SLURM job metadata from environment variables."""
import os


def get_slurm_info():
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        return None

    return {
        "job_id": job_id,
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURM_NODELIST"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "gpus": os.environ.get("SLURM_GPUS"),              # If allocated
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),  # If array job
    }
