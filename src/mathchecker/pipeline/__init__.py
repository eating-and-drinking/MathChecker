"""Inference pipeline components."""

__all__ = [
    "PedCoTPredictor",
    "LearnedRouterConfig",
    "LearnedSpecialistRouter",
    "RouteDecision",
    "RouterContext",
    "download_data_command",
    "evaluate_command",
    "rerun_failed_command",
    "run_command",
]


def __getattr__(name: str):
    if name == "PedCoTPredictor":
        from .predictor import PedCoTPredictor

        return PedCoTPredictor
    if name == "LearnedRouterConfig":
        from .router import LearnedRouterConfig

        return LearnedRouterConfig
    if name == "LearnedSpecialistRouter":
        from .router import LearnedSpecialistRouter

        return LearnedSpecialistRouter
    if name == "RouteDecision":
        from .router import RouteDecision

        return RouteDecision
    if name == "RouterContext":
        from .router import RouterContext

        return RouterContext
    if name in {"download_data_command", "evaluate_command", "rerun_failed_command", "run_command"}:
        from .runner import download_data_command, evaluate_command, rerun_failed_command, run_command

        return {
            "download_data_command": download_data_command,
            "evaluate_command": evaluate_command,
            "rerun_failed_command": rerun_failed_command,
            "run_command": run_command,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
