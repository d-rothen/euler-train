"""Collect git repository metadata for code_ref.json."""
import subprocess


def get_code_ref() -> dict:
    """Return a dict matching the code_refs schema."""
    return {
        "repo_url": _git("config", "--get", "remote.origin.url"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit_sha": _git("rev-parse", "HEAD"),
        "is_dirty": _is_dirty(),
        "dirty_diff": _dirty_diff(),
        "commit_message": _git("log", "-1", "--format=%B"),
        "committed_at": _git("log", "-1", "--format=%aI"),
    }


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:
        return None


def _is_dirty() -> bool:
    porcelain = _git("status", "--porcelain")
    return porcelain is not None


def _dirty_diff() -> str | None:
    if not _is_dirty():
        return None
    return _git("diff", "HEAD")
