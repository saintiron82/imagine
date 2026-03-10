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


def main():
    parser = argparse.ArgumentParser(description="WebDAV remote CLI")
    parser.add_argument('--test', type=str, help='Test connection (JSON config)')
    parser.add_argument('--list', type=str, help='List remote files (JSON config)')
    parser.add_argument('--folders', type=str, help='List subdirectories (JSON config with optional path)')
    args = parser.parse_args()

    try:
        if args.test:
            cmd_test(json.loads(args.test))
        elif args.list:
            cmd_list(json.loads(args.list))
        elif args.folders:
            cmd_folders(json.loads(args.folders))
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
