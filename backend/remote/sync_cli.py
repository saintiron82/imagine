"""
CLI entry point for WebDAV remote operations.
Called by Electron main process via subprocess.

Usage:
  python -m backend.remote.sync_cli --test '{"url":"...","username":"...","password":"...","remote_path":"/"}'
  python -m backend.remote.sync_cli --list '{"url":"...","username":"...","password":"...","remote_path":"/"}'
  python -m backend.remote.sync_cli --folders '{"url":"...","username":"...","password":"...","path":"/"}'
"""

import argparse
import json
import os
import sys

# Ensure project root is on sys.path for absolute imports
_project_root = os.environ.get('PYTHONPATH', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _emit(data: dict):
    """Write JSON line to stdout for Electron IPC."""
    print(json.dumps(data, ensure_ascii=False), flush=True)


def cmd_test(config: dict):
    """Test WebDAV connection."""
    from backend.remote.webdav_client import WebDAVClient

    client = WebDAVClient(
        base_url=config['url'],
        username=config['username'],
        password=config['password'],
        remote_path=config.get('remote_path', '/'),
        verify_ssl=config.get('verify_ssl', True),
    )
    result = client.test_connection()
    client.close()
    _emit(result)


def cmd_list(config: dict):
    """List remote files."""
    from backend.remote.webdav_client import WebDAVClient

    client = WebDAVClient(
        base_url=config['url'],
        username=config['username'],
        password=config['password'],
        remote_path=config.get('remote_path', '/'),
        verify_ssl=config.get('verify_ssl', True),
    )
    files = client.list_files_recursive()
    client.close()

    _emit({
        "success": True,
        "file_count": len(files),
        "files": [
            {"path": f.relative_path, "size": f.size}
            for f in files
        ],
    })


def cmd_folders(config: dict):
    """List subdirectories under a given path."""
    from backend.remote.webdav_client import WebDAVClient

    client = WebDAVClient(
        base_url=config['url'],
        username=config['username'],
        password=config['password'],
        remote_path=config.get('remote_path', '/'),
        verify_ssl=config.get('verify_ssl', True),
    )
    # Browse a specific sub-path if provided
    browse_path = config.get('path')
    folders = client.list_folders(path=browse_path)
    client.close()

    _emit({
        "success": True,
        "folders": folders,
    })


def _webdav_thumb_path(source_id: str, remote_path: str):
    """Unique thumbnail path for a WebDAV remote file: {THUMB_DIR}/webdav_{source_id}_{hash}_thumb.png"""
    import hashlib
    from pathlib import Path
    THUMB_DIR = Path(__file__).parent.parent.parent / "output" / "thumbnails"
    path_hash = hashlib.md5(remote_path.encode('utf-8')).hexdigest()[:12]
    stem = Path(remote_path).stem
    return THUMB_DIR / f"webdav_{source_id}_{stem}_{path_hash}_thumb.png"


def cmd_thumbnail(config: dict):
    """Download remote file → generate thumbnail → delete temp → return path.
    If thumbnail already exists on disk, return immediately without downloading."""
    import tempfile
    import shutil
    from pathlib import Path, PurePosixPath
    from backend.remote.webdav_client import WebDAVClient

    file_path = config['file_path']  # e.g., "/shared/sub/hero.psd"
    source_id = config.get('source_id', 'unknown')

    # Check if thumbnail already exists on disk
    thumb_path = _webdav_thumb_path(source_id, file_path)
    if thumb_path.exists():
        print(f"[WebDAV Thumb] cache HIT: {file_path} → {thumb_path}", file=sys.stderr, flush=True)
        _emit({"success": True, "thumb_path": str(thumb_path)})
        return

    print(f"[WebDAV Thumb] cache MISS, downloading {file_path}...", file=sys.stderr, flush=True)

    client = WebDAVClient(
        base_url=config['url'],
        username=config['username'],
        password=config['password'],
        remote_path=config.get('remote_path', '/'),
        verify_ssl=config.get('verify_ssl', True),
    )

    file_name = PurePosixPath(file_path).name

    # 1. Download to temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix='imagine_thumb_'))
    local_path = temp_dir / file_name

    try:
        success = client.download_file(file_path, local_path)
        client.close()

        if not success:
            print(f"[WebDAV Thumb] download FAILED: {file_path}", file=sys.stderr, flush=True)
            _emit({"success": False, "message": "Download failed"})
            return

        print(f"[WebDAV Thumb] downloaded {file_path} ({local_path.stat().st_size} bytes)", file=sys.stderr, flush=True)

        # 2. Generate thumbnail → save to unique WebDAV path
        from backend.utils.thumbnail_generator import process_single
        img, _ = process_single(str(local_path), size=256)
        if img is not None:
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(thumb_path), format='PNG', optimize=True)

        if thumb_path.exists():
            print(f"[WebDAV Thumb] generated {thumb_path}", file=sys.stderr, flush=True)
            _emit({"success": True, "thumb_path": str(thumb_path)})
        else:
            print(f"[WebDAV Thumb] generation FAILED for {file_path}", file=sys.stderr, flush=True)
            _emit({"success": False, "message": "Thumbnail generation failed"})
    finally:
        # 3. Delete temp file (original PSD), thumbnail persists on disk
        shutil.rmtree(temp_dir, ignore_errors=True)


def cmd_list_dir(config: dict):
    """List files + folders in a specific directory (non-recursive)."""
    from backend.remote.webdav_client import WebDAVClient

    client = WebDAVClient(
        base_url=config['url'],
        username=config['username'],
        password=config['password'],
        remote_path=config.get('remote_path', '/'),
        verify_ssl=config.get('verify_ssl', True),
    )
    result = client.list_dir(path=config.get('path'))
    client.close()
    _emit({"success": True, **result})


def main():
    parser = argparse.ArgumentParser(description="WebDAV remote CLI")
    parser.add_argument('--test', type=str, help='Test connection (JSON config)')
    parser.add_argument('--list', type=str, help='List remote files (JSON config)')
    parser.add_argument('--folders', type=str, help='List subdirectories (JSON config with optional path)')
    parser.add_argument('--list-dir', type=str, help='List files + folders non-recursively (JSON config)')
    parser.add_argument('--thumbnail', type=str, help='Download remote file and generate thumbnail (JSON config)')
    args = parser.parse_args()

    try:
        if args.test:
            cmd_test(json.loads(args.test))
        elif args.list:
            cmd_list(json.loads(args.list))
        elif args.folders:
            cmd_folders(json.loads(args.folders))
        elif getattr(args, 'list_dir', None):
            cmd_list_dir(json.loads(args.list_dir))
        elif args.thumbnail:
            cmd_thumbnail(json.loads(args.thumbnail))
        else:
            parser.print_help()
            sys.exit(1)
    except json.JSONDecodeError as e:
        _emit({"success": False, "message": f"Invalid JSON: {e}"})
        sys.exit(1)
    except Exception as e:
        _emit({"success": False, "message": f"Error: {e}"})
        sys.exit(1)


if __name__ == '__main__':
    main()
