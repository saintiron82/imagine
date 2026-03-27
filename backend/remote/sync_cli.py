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


def _parse_file(local_path):
    """Phase P equivalent: run parser → return metadata dict."""
    from pathlib import Path

    ext = Path(local_path).suffix.lower()
    meta = {"file_name": Path(local_path).name, "format": ext}

    try:
        if ext == '.psd':
            from backend.parser.psd_parser import PSDParser
            if PSDParser.can_parse(str(local_path)):
                result = PSDParser().parse(str(local_path))
                if result.success and result.asset_meta:
                    return result.asset_meta.model_dump()
        else:
            from backend.parser.image_parser import ImageParser
            if ImageParser.can_parse(str(local_path)):
                result = ImageParser().parse(str(local_path))
                if result.success and result.asset_meta:
                    return result.asset_meta.model_dump()
    except Exception as e:
        print(f"[WebDAV Browse] parse error: {e}", file=sys.stderr, flush=True)

    # Fallback: basic metadata
    try:
        from PIL import Image
        with Image.open(str(local_path)) as img:
            meta['resolution'] = list(img.size)
        meta['file_size'] = Path(local_path).stat().st_size
    except Exception:
        meta['file_size'] = Path(local_path).stat().st_size
    return meta


def _generate_thumb(local_path, source_id, remote_path):
    """Generate thumbnail → save to unique WebDAV path."""
    from pathlib import Path
    thumb_path = _webdav_thumb_path(source_id, remote_path)
    if thumb_path.exists():
        return thumb_path
    try:
        from backend.utils.thumbnail_generator import process_single
        img, _ = process_single(str(local_path), size=256)
        if img is not None:
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(thumb_path), format='PNG', optimize=True)
            return thumb_path
    except Exception as e:
        print(f"[WebDAV Browse] thumb error: {e}", file=sys.stderr, flush=True)
    return None


def _compute_content_hash(file_path):
    """SHA256 content hash (file_size + first 8KB + last 8KB)."""
    import hashlib
    from pathlib import Path
    p = Path(file_path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(p, 'rb') as f:
        h.update(f.read(8192))
        if size > 8192:
            f.seek(-8192, 2)
            h.update(f.read(8192))
    return h.hexdigest()


def cmd_browse(config: dict):
    """
    Browse a WebDAV folder: PROPFIND → DB check → download+parse uncached.
    Streams JSON events to stdout for Electron IPC.

    Events:
      {"event":"cached", "files":[{path, name, extension, thumb_path}, ...]}
      {"event":"processing", "path":"...", "index":N, "total":M}
      {"event":"processed", "path":"...", "name":"...", "thumb_path":"..."}
      {"event":"error", "path":"...", "message":"..."}
      {"event":"done", "cached":N, "processed":M}
    """
    import tempfile
    import shutil
    from pathlib import Path, PurePosixPath
    from queue import Queue
    from threading import Thread
    from backend.remote.webdav_client import WebDAVClient

    source_id = config['source_id']
    browse_path = config['path']

    print(f"[WebDAV Browse] start: source={source_id}, path={browse_path}", file=sys.stderr, flush=True)

    # 1. PROPFIND — file list
    client = WebDAVClient(
        base_url=config['url'],
        username=config['username'],
        password=config['password'],
        remote_path=config.get('remote_path', '/'),
        verify_ssl=config.get('verify_ssl', True),
    )
    result = client.list_dir(path=browse_path)
    files = result.get('files', [])
    print(f"[WebDAV Browse] PROPFIND: {len(files)} files", file=sys.stderr, flush=True)

    if not files:
        _emit({"event": "done", "cached": 0, "processed": 0})
        client.close()
        return

    # 2. DB check — which files already have thumbnails?
    try:
        from backend.db.sqlite_client import SQLiteDB
        db = SQLiteDB()
    except Exception as e:
        print(f"[WebDAV Browse] DB open failed: {e}, will process all", file=sys.stderr, flush=True)
        db = None

    cached = []
    uncached = []
    for idx, f in enumerate(files):
        canonical = f"webdav://{source_id}{f['path']}"
        if db is not None:
            try:
                # 1. Exact path match
                row = db.conn.execute(
                    "SELECT thumbnail_url FROM files WHERE file_path = ? AND thumbnail_url IS NOT NULL",
                    (canonical,)
                ).fetchone()

                # 2. Fallback: match by original filename at end of path
                #    DB may have different path prefix (e.g. /발송작품/ vs /발송4/발송작품/)
                if not row:
                    # Use the WebDAV relative path tail (folder + filename)
                    path_tail = f['path']  # e.g. /발송작품/작품/세일러문/#30/1/226.psd
                    row = db.conn.execute(
                        "SELECT thumbnail_url FROM files WHERE file_path LIKE ? AND thumbnail_url IS NOT NULL LIMIT 1",
                        (f"%{path_tail}",)
                    ).fetchone()

                # 3. Last resort: filename + file_size
                if not row and f.get('size'):
                    row = db.conn.execute(
                        "SELECT thumbnail_url FROM files WHERE file_path LIKE ? AND file_size = ? AND thumbnail_url IS NOT NULL LIMIT 1",
                        (f"%/{f['name']}", f['size'])
                    ).fetchone()

                if row and row[0] and Path(row[0]).exists():
                    cached.append({"path": canonical, "name": f['name'],
                                   "extension": f['extension'], "thumb_path": row[0]})
                    continue
            except Exception:
                pass
        uncached.append(f)

    print(f"[WebDAV Browse] cached={len(cached)}, uncached={len(uncached)}", file=sys.stderr, flush=True)

    if cached:
        _emit({"event": "cached", "files": cached})

    # Skip uncached files — only show files with existing thumbnails
    # (direct thumbnail generation is local-only)
    if uncached:
        for f in uncached:
            canonical = f"webdav://{source_id}{f['path']}"
            _emit({"event": "skipped", "path": canonical, "name": f['name']})
        print(f"[WebDAV Browse] skipped {len(uncached)} uncached files (local-only thumbnails)", file=sys.stderr, flush=True)

    _emit({"event": "done", "cached": len(cached), "processed": 0, "skipped": len(uncached)})
    if db:
        db.close()
    client.close()
    return

    # 3. Producer-Consumer: download (background) + parse (main) — DISABLED
    download_queue = Queue(maxsize=2)

    def downloader():
        for f in uncached:
            if not f.get('path'):
                continue
            temp_dir = Path(tempfile.mkdtemp(prefix='imagine_browse_'))
            local_path = temp_dir / PurePosixPath(f['path']).name
            try:
                success = client.download_file(f['path'], local_path)
                download_queue.put((f, local_path, temp_dir, success))
            except Exception as e:
                print(f"[WebDAV Browse] download error: {f['path']} - {e}", file=sys.stderr, flush=True)
                download_queue.put((f, None, temp_dir, False))
        download_queue.put(None)  # sentinel
        client.close()

    dl_thread = Thread(target=downloader, daemon=True)
    dl_thread.start()

    processed = 0
    total = len(uncached)
    while True:
        item = download_queue.get()
        if item is None:
            break
        f, local_path, temp_dir, success = item
        canonical = f"webdav://{source_id}{f['path']}"

        _emit({"event": "processing", "path": canonical, "index": processed + 1, "total": total})

        try:
            if not success or local_path is None or not local_path.exists():
                _emit({"event": "error", "path": canonical, "message": "Download failed"})
                continue

            file_size = local_path.stat().st_size
            print(f"[WebDAV Browse] processing: {f['name']} ({file_size} bytes)", file=sys.stderr, flush=True)

            # Phase P: parse
            meta = _parse_file(local_path)

            # Thumbnail
            thumb_path = _generate_thumb(local_path, source_id, f['path'])
            if thumb_path:
                meta['thumbnail_url'] = str(thumb_path)

            # Content hash
            meta['content_hash'] = _compute_content_hash(local_path)

            # DB upsert + create high-priority job for V/VV/MV processing
            meta['file_path'] = canonical
            meta['storage_root'] = f"webdav://{source_id}"
            if db is not None:
                try:
                    fid = db.upsert_metadata(canonical, meta)
                    # Only create processing job if thumbnail exists
                    # (V/VV/MV all require the thumbnail image)
                    if thumb_path:
                        parsed_md = json.dumps({
                            "metadata": meta,
                            "thumb_path": str(thumb_path),
                            "mc_raw": None,
                        }, ensure_ascii=False, default=str)
                        phase_completed = json.dumps({
                            "parse": True, "vision": False, "embed": False,
                        })
                        db.conn.execute(
                            """INSERT INTO job_queue
                               (file_id, file_path, status, priority,
                                parse_status, parsed_metadata, phase_completed)
                               VALUES (?, ?, 'pending', 100, 'parsed', ?, ?)
                               ON CONFLICT DO NOTHING""",
                            (fid, canonical, parsed_md, phase_completed)
                        )
                    else:
                        print(f"[WebDAV Browse] skipping job for {f['name']}: no thumbnail",
                              file=sys.stderr, flush=True)
                    db.conn.commit()
                except Exception as e:
                    print(f"[WebDAV Browse] DB upsert/job error: {e}", file=sys.stderr, flush=True)

            processed += 1
            _emit({"event": "processed", "path": canonical, "name": f['name'],
                   "thumb_path": str(thumb_path) if thumb_path else None})
            print(f"[WebDAV Browse] done: {f['name']} ({processed}/{total})", file=sys.stderr, flush=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    _emit({"event": "done", "cached": len(cached), "processed": processed})
    if db:
        db.close()
    dl_thread.join(timeout=5)
    print(f"[WebDAV Browse] complete: {len(cached)} cached, {processed} processed", file=sys.stderr, flush=True)


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
    parser.add_argument('--browse', type=str, help='Browse folder: PROPFIND + DB check + download+parse uncached (JSON config)')
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
        elif args.browse:
            cmd_browse(json.loads(args.browse))
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
