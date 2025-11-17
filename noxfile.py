"""Automation sessions for linting, type-checking and security scanning.

These sessions use Nox to create reproducible virtual environments and run
tools such as Ruff (lint/format), MyPy (type checks) and Bandit (security).
"""

import nox

# Global options
nox.options.sessions = ("ruff", "mypy", "bandit")
nox.options.reuse_existing_virtualenvs = True
nox.options.default_venv_backend = "uv|virtualenv"

SILENT_DEFAULT = True
SILENT_CODE_MODIFIERS = False

# Targets
PACKAGE_LOCATION = "."
CODE_LOCATIONS = PACKAGE_LOCATION
PYTHON_VERSIONS = ["3.12", "3.13"]
PYPY3_VERSION = "pypy3"
LATEST_PYTHON = PYTHON_VERSIONS[-1]


@nox.session(python=PYTHON_VERSIONS, tags=["lint", "format"])
def ruff(session: nox.Session) -> None:
    """Run Ruff to lint the codebase and apply formatting."""
    args = session.posargs or (PACKAGE_LOCATION,)
    _install(session, "ruff==0.12.7")
    _run(session, "ruff", "check", *args)
    _run_code_modifier(session, "ruff", "format", *args)


@nox.session(python=PYTHON_VERSIONS, tags=["typecheck"])
def mypy(session: nox.Session) -> None:
    """Verify static types using MyPy."""
    args = session.posargs or (PACKAGE_LOCATION,)
    # Install the project with all dependencies so mypy can find type information
    _install(session, PACKAGE_LOCATION)
    _install(session, "mypy", "types-requests", "typing-extensions", "types-PyYAML")
    _run(session, "mypy", *args)


@nox.session(python=PYTHON_VERSIONS, tags=["security"])
def bandit(session: nox.Session) -> None:
    """Scan Python files for common security issues using Bandit."""
    args = session.posargs or (CODE_LOCATIONS,)
    _install(session, "bandit")
    _run(session, "bandit", *args)


def _install(session: nox.Session, *args: str) -> None:
    """Install pip packages into the active Nox session."""
    if args:
        session.install(*args)


def _run(
    session: nox.Session,
    target: str,
    *args: str,
    silent: bool = SILENT_DEFAULT,
) -> None:
    """Run a command within the Nox session with standard options."""
    session.run(target, *args, external=True, silent=silent)


def _run_code_modifier(session: nox.Session, target: str, *args: str) -> None:
    """Run a code-modifying command with a less silent default."""
    _run(session, target, *args, silent=SILENT_CODE_MODIFIERS)
