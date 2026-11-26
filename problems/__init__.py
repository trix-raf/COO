import importlib
import pkgutil
from problems.base import ProblemSpec

_PROBLEM_REGISTRY: dict[str, ProblemSpec] = {}


def register_problem(spec: ProblemSpec) -> ProblemSpec:
    """Mendaftarkan objek ProblemSpec secara otomatis ke registry global."""
    _PROBLEM_REGISTRY[spec.code.upper()] = spec
    return spec


def _discover_problems() -> None:
    """Memuat otomatis seluruh modul p*_*.py di dalam paket problems/."""
    if _PROBLEM_REGISTRY:
        return
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("p") and "_" in module_info.name:
            importlib.import_module(f"{__name__}.{module_info.name}")


def get_problem(code: str) -> ProblemSpec:
    _discover_problems()
    key = code.strip().upper()
    if key not in _PROBLEM_REGISTRY:
        available = ", ".join(list_problem_codes())
        raise ValueError(f"Kode masalah '{code}' tidak ditemukan. Tersedia: {available}")
    return _PROBLEM_REGISTRY[key]


def list_problem_codes() -> list[str]:
    _discover_problems()
    return sorted(_PROBLEM_REGISTRY.keys(), key=lambda c: int(c[1:]) if c[1:].isdigit() else c)


__all__ = ["ProblemSpec", "register_problem", "get_problem", "list_problem_codes"]