"""Fail-fast checks for the production Streamlit authentication runtime."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


STREAMLIT_VERSION = "1.64.0"
AUTHLIB_VERSION = "1.8.0"
MINIMUM_AUTHLIB_VERSION = "1.3.2"


class AuthDependencyError(RuntimeError):
    """Authentication cannot run with the installed production packages."""


def _release(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(part) for part in value.split(".")[:3])
    except ValueError as error:
        raise AuthDependencyError(f"Unrecognized authentication dependency version {value}") from error
    if len(parts) != 3:
        raise AuthDependencyError(f"Unrecognized authentication dependency version {value}")
    return parts


def validate_auth_dependencies() -> dict[str, str]:
    """Import and validate the exact reviewed auth runtime without network access."""
    try:
        streamlit = import_module("streamlit")
        import_module("authlib")
        installed_streamlit = version("streamlit")
        installed_authlib = version("Authlib")
    except (ImportError, PackageNotFoundError) as error:
        raise AuthDependencyError(
            "Streamlit authentication dependencies are missing; install production requirements"
        ) from error
    if installed_streamlit != STREAMLIT_VERSION:
        raise AuthDependencyError(
            f"Unsupported Streamlit version {installed_streamlit}; expected {STREAMLIT_VERSION}"
        )
    if _release(installed_authlib) < _release(MINIMUM_AUTHLIB_VERSION):
        raise AuthDependencyError(
            f"Authlib {installed_authlib} is too old; {MINIMUM_AUTHLIB_VERSION} or newer is required"
        )
    if installed_authlib != AUTHLIB_VERSION:
        raise AuthDependencyError(
            f"Unsupported Authlib version {installed_authlib}; expected {AUTHLIB_VERSION}"
        )
    if not callable(getattr(streamlit, "login", None)):
        raise AuthDependencyError("Installed Streamlit does not provide st.login")
    return {"streamlit": installed_streamlit, "Authlib": installed_authlib}


if __name__ == "__main__":
    installed = validate_auth_dependencies()
    print(f"Authentication runtime ready: Streamlit {installed['streamlit']}, Authlib {installed['Authlib']}")
