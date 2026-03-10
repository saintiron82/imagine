"""
WebDAV client for remote folder sync.
Uses raw requests with WebDAV PROPFIND/GET methods.
Synology NAS WebDAV Server compatible.
"""

import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import List, Optional
from urllib.parse import unquote, quote

import requests
import xml.etree.ElementTree as ET

logger = logging.getLogger("WebDAVClient")

SUPPORTED_EXTENSIONS = {'.psd', '.png', '.jpg', '.jpeg'}

# DAV XML namespace
DAV_NS = "DAV:"
DAV = f"{{{DAV_NS}}}"


@dataclass
class RemoteFileInfo:
    """Metadata for a single remote file."""
    remote_path: str        # Full URL path (e.g., "/folder/sub/image.psd")
    relative_path: str      # Relative to source root (e.g., "sub/image.psd")
    size: int               # Content-Length
    last_modified: str       # Last-Modified header value
    etag: Optional[str]      # ETag header (if available)


class WebDAVClient:
    """
    WebDAV client for listing and downloading files from a remote server.
    Designed for Synology NAS WebDAV (https://address:5006/folder).
    """

    def __init__(self, base_url: str, username: str, password: str,
                 remote_path: str = "/", verify_ssl: bool = True,
                 timeout: int = 30):
        """
        Args:
            base_url: WebDAV server URL (e.g., "https://192.168.1.100:5006")
            username: WebDAV username
            password: WebDAV password
            remote_path: Remote directory path (e.g., "/shared/assets")
            verify_ssl: Whether to verify SSL certificates
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.remote_path = '/' + remote_path.strip('/')
        self.auth = (username, password)
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.verify = self.verify_ssl

    def _url(self, path: str) -> str:
        """Build full URL from path."""
        # Encode path components but preserve /
        parts = path.split('/')
        encoded = '/'.join(quote(p, safe='') for p in parts)
        return f"{self.base_url}{encoded}"

    def _nfc(self, text: str) -> str:
        """Normalize Unicode to NFC (for Korean filenames)."""
        return unicodedata.normalize('NFC', text)

    def test_connection(self) -> dict:
        """
        Test WebDAV connectivity.

        Returns:
            {"success": bool, "message": str, "file_count": int}
        """
        try:
            entries = self._propfind(self.remote_path, depth=1)
            # Count supported files (exclude the directory itself)
            file_count = sum(
                1 for e in entries
                if not e['is_collection'] and self._is_supported(e['href'])
            )
            return {
                "success": True,
                "message": f"Connected. {file_count} supported files found in root.",
                "file_count": file_count,
            }
        except requests.exceptions.SSLError:
            return {
                "success": False,
                "message": "SSL certificate verification failed. Try disabling SSL verification.",
                "file_count": 0,
            }
        except requests.exceptions.ConnectionError as e:
            return {
                "success": False,
                "message": f"Connection failed: {e}",
                "file_count": 0,
            }
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 401:
                msg = "Authentication failed. Check username/password."
            elif status == 404:
                msg = f"Path not found: {self.remote_path}"
            else:
                msg = f"HTTP error {status}: {e}"
            return {"success": False, "message": msg, "file_count": 0}
        except Exception as e:
            return {
                "success": False,
                "message": f"Unexpected error: {e}",
                "file_count": 0,
            }

    def list_files_recursive(self) -> List[RemoteFileInfo]:
        """
        Recursively list all supported image files under remote_path.
        Uses PROPFIND with Depth:1 iteratively (DFS).

        Returns:
            List of RemoteFileInfo for supported extensions only.
        """
        result = []
        dirs_to_visit = [self.remote_path]
        visited = set()

        while dirs_to_visit:
            current_dir = dirs_to_visit.pop()
            if current_dir in visited:
                continue
            visited.add(current_dir)

            try:
                entries = self._propfind(current_dir, depth=1)
            except Exception as e:
                logger.warning(f"Failed to list {current_dir}: {e}")
                continue

            for entry in entries:
                href = entry['href']

                # Skip self-reference
                if self._paths_equal(href, current_dir):
                    continue

                if entry['is_collection']:
                    dirs_to_visit.append(href)
                elif self._is_supported(href):
                    rel = self._relative_path(href)
                    if rel:
                        result.append(RemoteFileInfo(
                            remote_path=href,
                            relative_path=rel,
                            size=entry.get('size', 0),
                            last_modified=entry.get('last_modified', ''),
                            etag=entry.get('etag'),
                        ))

        logger.info(f"Found {len(result)} supported files on remote")
        return result

    def download_file(self, remote_path: str, local_path: Path,
                      expected_size: Optional[int] = None) -> bool:
        """
        Download a single file via GET with streaming.
        Creates parent directories as needed.
        Uses atomic write (*.downloading → rename).

        Returns:
            True on success, False on failure.
        """
        url = self._url(remote_path)
        tmp_path = local_path.with_suffix(local_path.suffix + '.downloading')

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            resp = self.session.get(url, stream=True, timeout=self.timeout)
            resp.raise_for_status()

            downloaded = 0
            with open(tmp_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

            # Verify size if expected
            if expected_size and downloaded != expected_size:
                logger.warning(
                    f"Size mismatch for {remote_path}: "
                    f"expected {expected_size}, got {downloaded}"
                )

            # Atomic rename
            tmp_path.replace(local_path)
            return True

        except Exception as e:
            logger.error(f"Download failed for {remote_path}: {e}")
            # Clean up partial file
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    # ── Internal helpers ──────────────────────────────────────

    def _propfind(self, path: str, depth: int = 1) -> list:
        """
        Send PROPFIND request, parse multistatus XML response.

        Returns:
            List of dicts: {href, is_collection, size, last_modified, etag}
        """
        url = self._url(path)
        headers = {
            'Depth': str(depth),
            'Content-Type': 'application/xml; charset=utf-8',
        }
        body = '''<?xml version="1.0" encoding="utf-8"?>
<d:propfind xmlns:d="DAV:">
  <d:prop>
    <d:resourcetype/>
    <d:getcontentlength/>
    <d:getlastmodified/>
    <d:getetag/>
  </d:prop>
</d:propfind>'''

        resp = self.session.request(
            'PROPFIND', url, headers=headers, data=body,
            timeout=self.timeout
        )
        resp.raise_for_status()

        return self._parse_multistatus(resp.text)

    def _parse_multistatus(self, xml_text: str) -> list:
        """Parse WebDAV multistatus XML response."""
        results = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse PROPFIND response: {e}")
            return results

        for response in root.findall(f'{DAV}response'):
            href_el = response.find(f'{DAV}href')
            if href_el is None or href_el.text is None:
                continue

            href = self._nfc(unquote(href_el.text))
            # Normalize trailing slashes for directories
            propstat = response.find(f'{DAV}propstat')
            if propstat is None:
                continue

            prop = propstat.find(f'{DAV}prop')
            if prop is None:
                continue

            # Check if collection (directory)
            rt = prop.find(f'{DAV}resourcetype')
            is_collection = (
                rt is not None and
                rt.find(f'{DAV}collection') is not None
            )

            # Get properties
            size_el = prop.find(f'{DAV}getcontentlength')
            size = int(size_el.text) if size_el is not None and size_el.text else 0

            lm_el = prop.find(f'{DAV}getlastmodified')
            last_modified = lm_el.text if lm_el is not None and lm_el.text else ''

            etag_el = prop.find(f'{DAV}getetag')
            etag = etag_el.text.strip('"') if etag_el is not None and etag_el.text else None

            results.append({
                'href': href,
                'is_collection': is_collection,
                'size': size,
                'last_modified': last_modified,
                'etag': etag,
            })

        return results

    def _is_supported(self, path: str) -> bool:
        """Check if file extension is supported."""
        ext = PurePosixPath(path).suffix.lower()
        return ext in SUPPORTED_EXTENSIONS

    def _relative_path(self, href: str) -> Optional[str]:
        """
        Convert absolute href to path relative to remote_path.
        Returns None if href is not under remote_path.
        """
        # Normalize both paths
        href_clean = href.rstrip('/')
        root_clean = self.remote_path.rstrip('/')

        if not href_clean.startswith(root_clean):
            return None

        rel = href_clean[len(root_clean):].lstrip('/')
        return self._nfc(rel) if rel else None

    def _paths_equal(self, path1: str, path2: str) -> bool:
        """Compare two paths ignoring trailing slashes."""
        return path1.rstrip('/') == path2.rstrip('/')
