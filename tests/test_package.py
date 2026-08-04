from importlib.metadata import version

import euler_train


def test_public_version_matches_distribution_metadata() -> None:
    assert euler_train.__version__ == version("euler-train")
