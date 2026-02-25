"""
Unified CLI entry point for all backend functionality.

When bundled with PyInstaller, this becomes the single executable (backend_cli.exe)
that Electron spawns for all Python operations.

Usage:
    backend_cli.exe worker-ipc
    backend_cli.exe search-daemon
    backend_cli.exe pipeline --discover "C:\\assets"
    backend_cli.exe server --port 8000
    backend_cli.exe stats
    ... etc

In dev mode, Electron uses `python -m backend.xxx` directly.
In packaged mode, Electron uses `backend_cli.exe <subcommand>`.
"""

import sys
import os
import io
import argparse
from pathlib import Path

# ── Ensure project root is in sys.path ──────────────────────────────
# When running from PyInstaller bundle, _MEIPASS is the temp extraction dir.
# When running from source, use the parent of this file.
if getattr(sys, 'frozen', False):
    # PyInstaller bundle
    BASE_DIR = Path(sys._MEIPASS)
else:
    # Running from source
    BASE_DIR = Path(__file__).parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── Force UTF-8 on Windows ──────────────────────────────────────────
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    if hasattr(sys.stdin, 'buffer'):
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)


# ── Subcommand handlers (lazy imports) ──────────────────────────────

def cmd_worker_ipc(args):
    """Worker IPC bridge — stdin/stdout JSON protocol for Electron."""
    from backend.worker.worker_ipc import main as worker_main
    worker_main()


def cmd_search_daemon(args):
    """Search daemon — persistent process with model caching."""
    sys.argv = ['api_search.py', '--daemon']
    from backend.api_search import main as search_main
    search_main()


def cmd_pipeline(args):
    """Pipeline — file processing (parse, vision, embed)."""
    argv = ['ingest_engine.py']
    if args.file:
        argv.extend(['--file', args.file])
    if args.files:
        argv.extend(['--files', args.files])
    if args.discover:
        argv.extend(['--discover', args.discover])
    if args.watch:
        argv.extend(['--watch', args.watch])
    if args.no_skip:
        argv.append('--no-skip')
    sys.argv = argv
    from backend.pipeline.ingest_engine import main as pipeline_main
    pipeline_main()


def cmd_server(args):
    """FastAPI server."""
    import uvicorn
    uvicorn.run(
        'backend.server.app:app',
        host=args.host,
        port=args.port,
        workers=1,
        log_level='info',
    )


def cmd_stats(args):
    """Database stats — returns JSON to stdout."""
    from backend.api_stats import main as stats_main
    stats_main()


def cmd_incomplete_stats(args):
    """Incomplete file stats."""
    from backend.api_incomplete_stats import main as incomplete_main
    incomplete_main()


def cmd_folder_stats(args):
    """Folder stats."""
    sys.argv = ['api_folder_stats.py', args.folder]
    from backend.api_folder_stats import main as folder_main
    folder_main()


def cmd_queue(args):
    """Job queue operations."""
    argv = ['api_queue.py', args.cmd]
    if args.data:
        argv.append(args.data)
    sys.argv = argv
    from backend.api_queue import main as queue_main
    queue_main()


def cmd_thumbnail(args):
    """Thumbnail generation."""
    argv = ['thumbnail_generator.py']
    if args.batch:
        argv.extend(['--batch', args.batch])
    elif args.file:
        argv.append(args.file)
    argv.extend(['--size', str(args.size)])
    if args.return_paths:
        argv.append('--return-paths')
    sys.argv = argv
    from backend.utils.thumbnail_generator import main as thumb_main
    thumb_main()


def cmd_installer(args):
    """Installer — check/install/download-model."""
    argv = ['installer.py']
    if args.check:
        argv.append('--check')
    if args.install:
        argv.append('--install')
    if args.download_model:
        argv.append('--download-model')
    sys.argv = argv
    from backend.setup.installer import main as installer_main
    installer_main()


def cmd_metadata_update(args):
    """Metadata update."""
    from backend.api_metadata_update import main as meta_main
    meta_main()


def cmd_relink(args):
    """Relink files."""
    argv = ['api_relink.py']
    if args.package:
        argv.extend(['--package', args.package])
    if args.folder:
        argv.extend(['--folder', args.folder])
    if args.dry_run:
        argv.append('--dry-run')
    if args.delete_missing:
        argv.append('--delete-missing')
    sys.argv = argv
    from backend.api_relink import main as relink_main
    relink_main()


def cmd_sync(args):
    """Sync folder — compare disk vs DB."""
    argv = ['api_sync.py', '--folder', args.folder]
    if args.apply_moves:
        argv.append('--apply-moves')
    if args.delete_missing:
        argv.extend(['--delete-missing', args.delete_missing])
    sys.argv = argv
    from backend.api_sync import main as sync_main
    sync_main()


def cmd_export(args):
    """Export database."""
    sys.argv = ['api_export.py', '--output', args.output]
    from backend.api_export import main as export_main
    export_main()


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='backend_cli',
        description='Imagine backend unified CLI'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # Worker IPC
    sub.add_parser('worker-ipc', help='Worker IPC bridge')

    # Search Daemon
    sub.add_parser('search-daemon', help='Search daemon')

    # Pipeline
    p = sub.add_parser('pipeline', help='File processing pipeline')
    p.add_argument('--file', help='Single file path')
    p.add_argument('--files', help='JSON array of file paths')
    p.add_argument('--discover', help='Directory to discover')
    p.add_argument('--watch', help='Directory to watch')
    p.add_argument('--no-skip', action='store_true', help='Disable smart skip')

    # Server
    p = sub.add_parser('server', help='FastAPI server')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--host', default='0.0.0.0')

    # Stats
    sub.add_parser('stats', help='Database stats')
    sub.add_parser('incomplete-stats', help='Incomplete file stats')
    p = sub.add_parser('folder-stats', help='Folder stats')
    p.add_argument('folder', help='Folder path')

    # Queue
    p = sub.add_parser('queue', help='Job queue operations')
    p.add_argument('cmd', help='Queue command')
    p.add_argument('data', nargs='?', help='JSON data')

    # Thumbnail
    p = sub.add_parser('thumbnail', help='Thumbnail generation')
    p.add_argument('--batch', help='JSON batch data')
    p.add_argument('--file', help='Single file path')
    p.add_argument('--size', type=int, default=256, help='Thumbnail size')
    p.add_argument('--return-paths', action='store_true')

    # Installer
    p = sub.add_parser('installer', help='Installation utilities')
    p.add_argument('--check', action='store_true')
    p.add_argument('--install', action='store_true')
    p.add_argument('--download-model', action='store_true')

    # Metadata
    sub.add_parser('metadata-update', help='Update metadata')

    # Relink
    p = sub.add_parser('relink', help='Relink files')
    p.add_argument('--package')
    p.add_argument('--folder')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--delete-missing', action='store_true')

    # Sync
    p = sub.add_parser('sync', help='Sync folder')
    p.add_argument('--folder', required=True)
    p.add_argument('--apply-moves', action='store_true')
    p.add_argument('--delete-missing', help='Comma-separated file IDs to delete')

    # Export
    p = sub.add_parser('export', help='Export database')
    p.add_argument('--output', required=True)

    args = parser.parse_args()

    # Dispatch to handler
    handlers = {
        'worker-ipc': cmd_worker_ipc,
        'search-daemon': cmd_search_daemon,
        'pipeline': cmd_pipeline,
        'server': cmd_server,
        'stats': cmd_stats,
        'incomplete-stats': cmd_incomplete_stats,
        'folder-stats': cmd_folder_stats,
        'queue': cmd_queue,
        'thumbnail': cmd_thumbnail,
        'installer': cmd_installer,
        'metadata-update': cmd_metadata_update,
        'relink': cmd_relink,
        'sync': cmd_sync,
        'export': cmd_export,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
