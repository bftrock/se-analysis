"""Where data lives on disk, so notebooks do not hardcode it.

Three kinds of anchor: repository-relative, so a notebook finds committed data
regardless of the kernel's CWD; user-relative, so a notebook reading a file
someone just downloaded runs unchanged on anyone's machine; and configured, for
the projects folder, whose location Windows cannot be asked about reliably.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


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


def _find_documents() -> Path:
    """The user's Documents folder, wherever Windows has actually been told to put it.

    Same story as Downloads: OneDrive's Known Folder Move relocates Documents on
    plenty of machines, so ask the registry before assuming the user profile.
    """
    conventional = Path.home() / "Documents"

    if sys.platform == "win32":
        import winreg

        shell_folders = (
            r"Software\Microsoft\Windows\CurrentVersion\Explorer" r"\User Shell Folders"
        )
        # "Personal" is the historical value name for Documents and the GUID is the
        # modern one; machines usually have both, but do not count on either.
        documents_names = ("Personal", "{F42EE2D3-909F-4907-8871-4C22FC0BF756}")
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, shell_folders) as key:
                for name in documents_names:
                    try:
                        registered, _ = winreg.QueryValueEx(key, name)
                    except OSError:
                        continue  # this name is not registered; try the other
                    # the stored value may still contain %USERPROFILE%
                    return Path(os.path.expandvars(registered))
        except OSError:
            pass  # no access; the convention is the best guess

    return conventional


def _find_projects() -> Path:
    """Where the Projects folder lives, which is genuinely machine-specific.

    SE_PROJECTS_DIR wins if set; it belongs in .env next to the NRG Cloud
    credentials. Unlike Downloads, this cannot be settled from the registry:
    Windows tracks exactly one Documents folder, so on a machine with both a
    personal and a work OneDrive signed in, the one it reports is a coin flip
    against the one holding your projects.

    The configured value may name environment variables -- %OneDriveCommercial%
    is the work OneDrive on any machine, whatever the tenant folder is called.
    """
    configured = os.getenv("SE_PROJECTS_DIR")
    if configured:
        return Path(os.path.expandvars(configured)).expanduser()

    return _find_documents() / "Projects"


REPO_ROOT = _find_repo_root()
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# .env lives at the repository root; look there rather than trusting the kernel's
# CWD. This runs at import, before a notebook gets its own load_dotenv() in, and
# does not override variables the environment already set -- those still win.
load_dotenv(REPO_ROOT / ".env")

DOWNLOADS_DIR = _find_downloads()
PROJECTS_DIR = _find_projects()


def project(*parts: str) -> Path:
    """Absolute path under PROJECTS_DIR, e.g. project("Keene-Branch_FL", "data.dat").

    Raises if the projects folder itself is missing, which is almost always a
    machine that needs SE_PROJECTS_DIR set rather than a genuinely absent file.
    """
    if not PROJECTS_DIR.is_dir():
        if os.getenv("SE_PROJECTS_DIR"):
            detail = "SE_PROJECTS_DIR points there, and there is no such folder."
        else:
            detail = (
                "SE_PROJECTS_DIR is unset, so this is a guess from the Documents "
                "folder Windows reports -- which is the wrong one on a machine "
                "with more than one OneDrive account signed in."
            )
        raise RuntimeError(
            f"No projects folder at {PROJECTS_DIR}. {detail} Set SE_PROJECTS_DIR "
            "in .env to wherever your Projects folder actually lives; see "
            ".env.example."
        )

    return PROJECTS_DIR.joinpath(*parts)


def download(*parts: str) -> Path:
    """Absolute path under the user's Downloads, e.g. download("MET 1 Modbus", "x.txt").

    For files that are not committed to the repository: whoever has the file in
    their own Downloads can run the notebook unchanged.
    """
    return DOWNLOADS_DIR.joinpath(*parts)
