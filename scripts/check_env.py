from __future__ import annotations

from importlib.metadata import version as package_version
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
(ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
(ROOT / ".cache").mkdir(parents=True, exist_ok=True)

import gymnasium  # noqa: E402
import matplotlib  # noqa: E402
import minigrid  # noqa: E402
import networkx  # noqa: E402
import numpy  # noqa: E402
import pandas  # noqa: E402
import scipy  # noqa: E402
import seaborn  # noqa: E402
import sklearn  # noqa: E402
import tqdm  # noqa: E402


def main() -> None:
    versions = {
        "numpy": numpy.__version__,
        "matplotlib": matplotlib.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "seaborn": getattr(seaborn, "__version__", package_version("seaborn")),
        "tqdm": tqdm.__version__,
        "rich": package_version("rich"),
        "gymnasium": gymnasium.__version__,
        "minigrid": minigrid.__version__,
        "networkx": networkx.__version__,
    }
    for name, pkg_version in versions.items():
        print(f"{name}: {pkg_version}")
    print("environment ok")


if __name__ == "__main__":
    main()
