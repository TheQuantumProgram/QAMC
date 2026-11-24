"""QAMC checker package."""

from qamc.checker import CheckerRunResult, run_model, run_qam_file
from qamc.loader import load_qam_file, load_qam_string, normalize_qam_data

__all__ = [
    "__version__",
    "CheckerRunResult",
    "load_qam_file",
    "load_qam_string",
    "normalize_qam_data",
    "run_model",
    "run_qam_file",
]

__version__ = "0.1.0"
