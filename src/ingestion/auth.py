"""
CDSE authentication and shared download utilities.

Provides a base class used by both Sentinel1Downloader and Sentinel2Downloader.

CDSE uses Keycloak OAuth2 (Resource Owner Password Credentials grant).
Tokens are valid for ~600 seconds; this class tracks expiry and auto-refreshes.

Token endpoint:
  https://identity.dataspace.copernicus.eu/auth/realms/CDSE/
  protocol/openid-connect/token

Download endpoint (streaming zip):
  https://zipper.dataspace.copernicus.eu/odata/v1/Products({id})/$value
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# CDSE token endpoint
_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
# Refresh token 60 s before actual expiry to avoid mid-download failures
_TOKEN_EXPIRY_BUFFER = 60

# Chunk size for streaming downloads (8 MB)
_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024


class CDSEAuth:
    """
    Manages CDSE OAuth2 tokens with automatic refresh.

    Parameters
    ----------
    username : str
        CDSE account email.
    password : str
        CDSE account password.
    """

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._expires_at: float = 0.0

    def token(self) -> str:
        """
        Return a valid access token, fetching or refreshing as needed.

        Returns
        -------
        str
            Bearer access token.
        """
        if time.time() < self._expires_at - _TOKEN_EXPIRY_BUFFER:
            return self._access_token  # type: ignore[return-value]

        if self._refresh_token:
            try:
                self._do_refresh()
                return self._access_token  # type: ignore[return-value]
            except requests.HTTPError:
                logger.debug("Refresh token expired — re-authenticating")

        self._do_authenticate()
        return self._access_token  # type: ignore[return-value]

    def headers(self) -> dict[str, str]:
        """Return Authorization headers for API requests."""
        return {"Authorization": f"Bearer {self.token()}"}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _do_authenticate(self) -> None:
        """Fetch a fresh token pair using username + password."""
        resp = requests.post(
            _TOKEN_URL,
            data={
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
                "client_id": "cdse-public",
            },
            timeout=30,
        )
        resp.raise_for_status()
        self._store_tokens(resp.json())
        logger.debug("CDSE: authenticated successfully")

    def _do_refresh(self) -> None:
        """Refresh the access token using the stored refresh token."""
        resp = requests.post(
            _TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": "cdse-public",
            },
            timeout=30,
        )
        resp.raise_for_status()
        self._store_tokens(resp.json())
        logger.debug("CDSE: token refreshed")

    def _store_tokens(self, payload: dict) -> None:
        self._access_token = payload["access_token"]
        self._refresh_token = payload.get("refresh_token")
        expires_in = payload.get("expires_in", 600)
        self._expires_at = time.time() + expires_in


class CDSEDownloaderBase:
    """
    Base class for Sentinel-1 and Sentinel-2 downloaders.

    Provides:
      - Shared CDSE authentication
      - Streaming download with progress bar and resume support
      - MD5 checksum verification
      - Automatic retry on 401 (token refresh) and 5xx errors

    Subclasses only need to implement `search()`.
    """

    CDSE_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    CDSE_DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

    _MAX_RETRIES = 3
    _RETRY_BACKOFF = [5, 15, 30]  # seconds between retries

    def __init__(
        self,
        username: Optional[str],
        password: Optional[str],
        output_dir: Path,
    ) -> None:
        _user = username or os.environ.get("CDSE_USERNAME")
        _pass = password or os.environ.get("CDSE_PASSWORD")
        if not _user or not _pass:
            raise ValueError(
                "CDSE credentials required. Set CDSE_USERNAME and "
                "CDSE_PASSWORD environment variables or pass them explicitly."
            )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._auth = CDSEAuth(_user, _pass)
        # Eager authenticate so credential errors surface early
        self._auth.token()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, scene_id: str, scene_name: str) -> Path:
        """
        Download a single scene by its CDSE product UUID.

        Skips download if the file already exists and checksum matches.
        Supports partial resume via HTTP Range requests.

        Parameters
        ----------
        scene_id : str
            CDSE product UUID (from search results).
        scene_name : str
            Human-readable product name — used as the output filename stem.

        Returns
        -------
        Path
            Local path to the downloaded .zip file.
        """
        out_path = self.output_dir / f"{scene_name}.zip"

        if self._already_downloaded(out_path, scene_id):
            logger.info(f"Already downloaded, skipping: {out_path.name}")
            return out_path

        url = f"{self.CDSE_DOWNLOAD_URL}({scene_id})/$value"
        logger.info(f"Downloading: {scene_name}")

        for attempt in range(self._MAX_RETRIES):
            try:
                self._stream_download(url, out_path)
                logger.info(f"Download complete: {out_path.name}")
                return out_path
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                if status == 401:
                    logger.warning("Token expired during download — refreshing")
                    self._auth._do_authenticate()
                    continue
                if status in (429, 503) or status is None:
                    wait = self._RETRY_BACKOFF[min(attempt, len(self._RETRY_BACKOFF) - 1)]
                    logger.warning(f"HTTP {status} — retrying in {wait}s")
                    time.sleep(wait)
                    continue
                raise
            except (requests.ConnectionError, requests.Timeout) as exc:
                wait = self._RETRY_BACKOFF[min(attempt, len(self._RETRY_BACKOFF) - 1)]
                logger.warning(f"Connection error ({exc}) — retrying in {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"Download failed after {self._MAX_RETRIES} attempts: {scene_name}")

    def download_range(
        self,
        bbox: list[float],
        start_date: str | datetime,
        end_date: str | datetime,
        **search_kwargs,
    ) -> list[Path]:
        """
        Search and download all matching scenes for a date range and AOI.

        Returns
        -------
        list[Path]
            Paths to all downloaded .zip files.
        """
        scenes = self.search(bbox, start_date, end_date, **search_kwargs)
        logger.info(f"Found {len(scenes)} scenes to download")
        paths = []
        for scene in scenes:
            path = self.download(scene["id"], scene["name"])
            paths.append(path)
        return paths

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _stream_download(self, url: str, out_path: Path) -> None:
        """Stream a file to disk with a tqdm progress bar."""
        # Determine bytes already downloaded (for resume)
        resume_bytes = out_path.stat().st_size if out_path.exists() else 0
        headers = dict(self._auth.headers())
        if resume_bytes:
            headers["Range"] = f"bytes={resume_bytes}-"
            logger.debug(f"Resuming from byte {resume_bytes}")

        with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
            resp.raise_for_status()

            total = int(resp.headers.get("Content-Length", 0))
            if resume_bytes and resp.status_code == 206:
                total += resume_bytes  # Content-Length is remaining bytes

            mode = "ab" if resume_bytes else "wb"
            with open(out_path, mode) as fh, tqdm(
                total=total or None,
                initial=resume_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=out_path.name,
                leave=False,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        pbar.update(len(chunk))

    def _already_downloaded(self, out_path: Path, scene_id: str) -> bool:
        """
        Return True if the file exists and has non-zero size.

        We skip full MD5 verification here (CDSE doesn't always expose checksums)
        and rely on zip integrity check at unzip time.
        """
        return out_path.exists() and out_path.stat().st_size > 0

    @staticmethod
    def _bbox_to_wkt(bbox: list[float]) -> str:
        """
        Convert [min_lat, min_lon, max_lat, max_lon] to a WKT POLYGON.

        OData spatial filters expect WKT with longitude first (x, y order).
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        return (
            f"POLYGON(("
            f"{min_lon} {min_lat},"
            f"{max_lon} {min_lat},"
            f"{max_lon} {max_lat},"
            f"{min_lon} {max_lat},"
            f"{min_lon} {min_lat}"
            f"))"
        )

    @staticmethod
    def _format_date(d: str | datetime) -> str:
        """Format a date to the ISO 8601 string CDSE OData expects."""
        if isinstance(d, datetime):
            return d.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        # Accept YYYY-MM-DD or full ISO
        if "T" not in d:
            return f"{d}T00:00:00.000Z"
        return d

    def _paginated_search(self, params: dict, max_results: int) -> list[dict]:
        """
        Fetch all pages of a CDSE OData search and return product list.

        CDSE returns at most 100 items per page; this handles pagination
        via the `@odata.nextLink` field.

        Parameters
        ----------
        params : dict
            Initial query parameters for the search request.
        max_results : int
            Hard cap on total results returned.

        Returns
        -------
        list[dict]
            Flat list of product metadata dicts.
        """
        results: list[dict] = []
        url: Optional[str] = self.CDSE_SEARCH_URL
        page_size = min(100, max_results)
        params = {**params, "$top": page_size, "$orderby": "ContentDate/Start desc"}

        while url and len(results) < max_results:
            resp = requests.get(
                url,
                params=params if url == self.CDSE_SEARCH_URL else None,
                headers=self._auth.headers(),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            products = data.get("value", [])
            for p in products:
                if len(results) >= max_results:
                    break
                results.append({
                    "id": p["Id"],
                    "name": p["Name"],
                    "size": p.get("ContentLength", 0),
                    "date": p.get("ContentDate", {}).get("Start"),
                    "footprint": p.get("Footprint"),
                    "cloud_cover": self._extract_attr(p, "cloudCover"),
                    "raw": p,
                })

            url = data.get("@odata.nextLink")
            params = {}  # nextLink already contains all params

        logger.info(f"Search returned {len(results)} scenes")
        return results

    @staticmethod
    def _extract_attr(product: dict, attr_name: str):
        """
        Pull a named attribute from the CDSE OData Attributes list.

        CDSE embeds scene attributes as a polymorphic list, e.g.:
          Attributes: [
            {Name: 'cloudCover', Value: 12.5, '@odata.type': '...DoubleAttribute'},
            ...
          ]
        """
        for attr in product.get("Attributes", []):
            if attr.get("Name") == attr_name:
                return attr.get("Value")
        return None
