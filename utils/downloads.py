# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Download utils."""

import logging
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def is_url(url, check=True):
    """    Determines if a string is a URL and optionally checks its existence
    online, returning a boolean.

    Args:
        url (str): The input string to be checked as a URL.
        check (bool?): Flag to indicate whether to check the existence of the URL online.
            Defaults to True.

    Returns:
        bool: True if the input is a valid URL and, if specified, exists online; False
            otherwise.
    """
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def gsutil_getsize(url=""):
    """    Returns the size in bytes of a file at a Google Cloud Storage URL using
    `gsutil du`.

    This function executes the `gsutil du` command to get the size of the
    file at the specified URL. If the command fails or the output is empty,
    it returns 0.

    Args:
        url (str): The Google Cloud Storage URL of the file.

    Returns:
        int: The size of the file in bytes.
    """
    output = subprocess.check_output(["gsutil", "du", url], shell=False, encoding="utf-8")
    return int(output.split()[0]) if output else 0


def url_getsize(url="https://ultralytics.com/images/bus.jpg"):
    """    Returns the size in bytes of a downloadable file at a given URL;
    defaults to -1 if not found.

    Args:
        url (str): The URL of the file to get the size from. Defaults to
            "https://ultralytics.com/images/bus.jpg".

    Returns:
        int: The size in bytes of the file at the given URL, or -1 if not found.
    """
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get("content-length", -1))


def curl_download(url, filename, *, silent: bool = False) -> bool:
    """    Download a file from a URL to a specified filename using curl.

    Args:
        url (str): The URL from which the file will be downloaded.
        filename (str): The name of the file to which the downloaded content will be saved.
        silent (bool?): If True, suppresses the progress meter and error messages. Defaults to
            False.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    silent_option = "sS" if silent else ""  # silent
    proc = subprocess.run(
        [
            "curl",
            "-#",
            f"-{silent_option}L",
            url,
            "--output",
            filename,
            "--retry",
            "9",
            "-C",
            "-",
        ]
    )
    return proc.returncode == 0


def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    """    Downloads a file from a URL (or alternate URL) to a specified path if
    the file size is above a minimum threshold.

    This function downloads a file from the specified URL to the provided
    file path. If the downloaded file is incomplete or below the minimum
    size threshold, it removes the incomplete download and retries the
    download from an alternate URL if provided.

    Args:
        file (str): The path where the downloaded file will be saved.
        url (str): The primary URL from which the file will be downloaded.
        url2 (str?): An alternate URL to download the file from if the primary download
            fails. Defaults to None.
        min_bytes (float?): The minimum size threshold in bytes for the downloaded file to be
            considered valid. Defaults to 1e0.
        error_msg (str?): Custom error message to display when the downloaded file is incomplete
            or below the minimum size. Defaults to "".
    """
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # remove partial downloads
        LOGGER.info(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        # curl download, retry and resume on fail
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info("")


def attempt_download(file, repo="ultralytics/yolov5", release="v7.0"):
    """    Downloads a file from GitHub release assets or via direct URL if not
    found locally, supporting backup versions.

    This function attempts to download a file from GitHub release assets or
    via a direct URL if the file is not found locally. It supports backup
    versions in case the specified file is not available.

    Args:
        file (str): The file to be downloaded.
        repo (str): The GitHub repository to download from. Defaults to
            "ultralytics/yolov5".
        release (str): The release version to download from. Defaults to "v7.0".

    Returns:
        str: The path to the downloaded file.
    """
    from utils.general import LOGGER

    def github_assets(repository, version="latest"):
        """Return GitHub repository tag and assets based on the specified version.

        This function takes a GitHub repository name and an optional version
        parameter to fetch the repository tag and associated assets from the
        GitHub API.

        Args:
            repository (str): The name of the GitHub repository.
            version (str?): The version or tag of the repository. Defaults to "latest".

        Returns:
            tuple: A tuple containing the repository tag (str) and a list of asset names
                (list of str).
        """

        # Return GitHub repo tag (i.e. 'v7.0') and assets (i.e. ['yolov5s.pt', 'yolov5m.pt', ...])
        if version != "latest":
            version = f"tags/{version}"  # i.e. tags/v7.0
        response = requests.get(f"https://api.github.com/repos/{repository}/releases/{version}").json()  # github api
        return response["tag_name"], [x["name"] for x in response["assets"]]  # tag, assets

    file = Path(str(file).strip().replace("'", ""))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = name.split("?")[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f"Found {url} locally at {file}")  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1e5)
            return file

        # GitHub assets
        assets = [f"yolov5{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "6", "-cls", "-seg")]  # default
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output("git tag", shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        if name in assets:
            file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag}",
            )

    return str(file)
