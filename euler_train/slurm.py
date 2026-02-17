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
        "num_nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        "ntasks": os.environ.get("SLURM_NTASKS"),
        "ntasks_per_node": os.environ.get("SLURM_NTASKS_PER_NODE"),
        "gpus_per_node": os.environ.get("SLURM_GPUS_PER_NODE"),
        "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "mem_per_cpu": os.environ.get("SLURM_MEM_PER_CPU"),
        "stdout_path": os.environ.get("SLURM_JOB_STDOUT"),  # --output log
        "stderr_path": os.environ.get("SLURM_JOB_STDERR"),  # --error log
        "submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
    }
