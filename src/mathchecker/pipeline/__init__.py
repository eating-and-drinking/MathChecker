"""Inference pipeline components.

The legacy `LearnedSpecialistRouter` / `RouteDecision` / `RouterContext`
exports have been removed as part of the PRISM refactor. Specialist routing
now lives in `mathchecker.prism.eig` (Expected Information Gain).
"""

__all__ = [
    "PedCoTPredictor",
    "download_data_command",
    "evaluate_command",
    "rerun_failed_command",
    "run_command",
]


def __getattr__(name: str):
    if name == "PedCoTPredictor":
        from .predictor import PedCoTPredictor

        return PedCoTPredictor
    if name in {"download_data_command", "evaluate_command", "rerun_failed_command", "run_command"}:
        from .runner import download_data_command, evaluate_command, rerun_failed_command, run_command

        return {
            "download_data_command": download_data_command,
            "evaluate_command": evaluate_command,
            "rerun_failed_command": rerun_failed_command,
            "run_command": run_command,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
