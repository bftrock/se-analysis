"""Where data lives on disk, so notebooks do not hardcode it.

Two kinds of anchor: repository-relative, so a notebook finds committed data
regardless of the kernel's CWD, and user-relative, so a notebook reading a file
someone just downloaded runs unchanged on anyone's machine.
"""

import os
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file until we find the directory holding pyproject.toml."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        "Could not locate the se-analysis repository root. This only works from an "
        "editable install; reinstall with `uv sync`."
    )


def _find_downloads() -> Path:
    """The user's Downloads folder, wherever Windows has actually been told to put it.

    Downloads can be moved off the user profile -- OneDrive's Known Folder Move,
    or a second drive -- so ask the registry before assuming. Falls back to the
    conventional location, which is also the answer everywhere but Windows.
    """
    conventional = Path.home() / "Downloads"

    if sys.platform == "win32":
        import winreg

        shell_folders = (
            r"Software\Microsoft\Windows\CurrentVersion\Explorer" r"\User Shell Folders"
        )
        downloads_guid = "{374DE290-123F-4565-9164-39C4925E467B}"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, shell_folders) as key:
                registered, _ = winreg.QueryValueEx(key, downloads_guid)
            # the stored value may still contain %USERPROFILE%
            return Path(os.path.expandvars(registered))
        except OSError:
            pass  # not registered, or no access; the convention is the best guess

    return conventional


REPO_ROOT = _find_repo_root()
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOWNLOADS_DIR = _find_downloads()
PROJECTS_DIR = Path("/mnt/jupyter_data/workspace")


def project(*parts: str) -> Path:
    """Absolute path under PROJECTS_DIR, e.g. project("Keene-Branch_FL", "data.dat")."""
    return PROJECTS_DIR.joinpath(*parts)


def download(*parts: str) -> Path:
    """Absolute path under the user's Downloads, e.g. download("MET 1 Modbus", "x.txt").

    For files that are not committed to the repository: whoever has the file in
    their own Downloads can run the notebook unchanged.
    """
    return DOWNLOADS_DIR.joinpath(*parts)
