"""Output-dir + config-snapshot helpers.

These mirror ``utils.utils.get_output_dir`` / ``copy_config_to_results`` exactly, but are
reimplemented here (4 trivial lines) rather than imported: importing ``utils.utils`` pulls
``quotientai``/``pandas`` at module load, and coupling the arena core to those heavy deps
contradicts the lean intent. Behaviour and the ``results/<type>/<timestamp>/`` layout are
identical, so a maintainer sees the same output convention.
"""

import logging
import os
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


def get_output_dir(output_dir: str, subdir: str = "arena") -> str:
    """Return ``<output_dir>/<subdir>/<timestamp>`` — same shape as the base's runs."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(output_dir, subdir, timestamp)


def copy_config_to_results(config_path: str, output_dir: str) -> None:
    """Snapshot the run config into the output dir (best-effort, never raises)."""
    try:
        if config_path and os.path.exists(config_path):
            shutil.copy2(config_path, output_dir)
        else:
            logger.info("No config file to snapshot (zero-config run)")
    except Exception as e:
        logger.error(f"Error copying config file: {e}")
