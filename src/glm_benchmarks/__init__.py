import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

if "profile" not in __builtins__:  # type: ignore
    __builtins__["profile"] = lambda x: x  # type: ignore
