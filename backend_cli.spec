# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Imagine backend_cli — unified backend executable.

Build:
    pyinstaller backend_cli.spec --noconfirm

Output:
    dist/backend_cli/backend_cli.exe  (Windows)
    dist/backend_cli/backend_cli      (macOS/Linux)
"""

import sys
import os
from pathlib import Path

block_cipher = None

# Platform-specific excludes
platform_excludes = [
    'tkinter', '_tkinter', 'unittest',
    'xmlrpc', 'pydoc', 'doctest',
]

if sys.platform == 'win32':
    platform_excludes += ['mlx', 'mlx_vlm', 'mlx_lm', 'vllm']
elif sys.platform == 'darwin':
    platform_excludes += ['vllm']

# Collect data files
datas = [
    ('config.yaml', '.'),
    ('backend/vision/domains', 'backend/vision/domains'),
    ('backend/db/sqlite_schema.sql', 'backend/db'),
    ('backend/db/sqlite_schema_auth.sql', 'backend/db'),
]

# Add migrations if they exist
if os.path.exists('backend/db/migrations'):
    datas.append(('backend/db/migrations', 'backend/db/migrations'))

a = Analysis(
    ['backend/backend_cli.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # ── Core backend modules ──
        'backend.worker.worker_ipc',
        'backend.worker.worker_daemon',
        'backend.worker.config',
        'backend.worker.worker_state',
        'backend.worker.schedule',
        'backend.worker.resource_monitor',
        'backend.worker.result_uploader',

        'backend.pipeline.ingest_engine',

        'backend.parser.base_parser',
        'backend.parser.psd_parser',
        'backend.parser.image_parser',
        'backend.parser.schema',
        'backend.parser.cleaner',

        'backend.vision.vision_factory',
        'backend.vision.analyzer',
        'backend.vision.ollama_adapter',
        'backend.vision.domain_loader',
        'backend.vision.schemas',
        'backend.vision.prompts',
        'backend.vision.repair',

        'backend.vector.siglip2_encoder',
        'backend.vector.text_embedding',

        'backend.db.sqlite_client',
        'backend.db.db_interface',
        'backend.db.write_queue',

        'backend.search.sqlite_search',
        'backend.search.rrf',
        'backend.search.query_decomposer',

        'backend.utils.config',
        'backend.utils.tier_config',
        'backend.utils.tier_compatibility',
        'backend.utils.platform_detector',
        'backend.utils.parent_watchdog',
        'backend.utils.win32_stdin',
        'backend.utils.meta_helpers',
        'backend.utils.thumbnail_generator',
        'backend.utils.adaptive_batch_controller',
        'backend.utils.gpu_detect',
        'backend.utils.memory_monitor',
        'backend.utils.dhash',
        'backend.utils.content_hash',

        'backend.server.app',
        'backend.server.deps',
        'backend.server.auth.router',
        'backend.server.auth.jwt',
        'backend.server.auth.schemas',
        'backend.server.routers.workers',
        'backend.server.routers.pipeline',
        'backend.server.routers.worker_setup',
        'backend.server.routers.classification',
        'backend.server.routers.database',
        'backend.server.queue.manager',

        'backend.setup.installer',

        'backend.api_search',
        'backend.api_stats',
        'backend.api_incomplete_stats',
        'backend.api_folder_stats',
        'backend.api_queue',
        'backend.api_metadata_update',
        'backend.api_relink',
        'backend.api_sync',
        'backend.api_export',

        # ── Third-party hidden imports ──
        'sqlite_vec',
        'psd_tools',
        'PIL',
        'PIL._imaging',
        'numpy',
        'torch',
        'torch._C',
        'torch.utils._python_dispatch',
        'transformers',
        'accelerate',
        'pydantic',
        'pydantic.deprecated.decorator',
        'yaml',
        'requests',
        'psutil',
        'watchdog',
        'watchdog.observers',
        'tqdm',
        'exifread',
        'deep_translator',
        'dotenv',
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'starlette',
        'jose',
        'passlib',
        'multipart',
    ],
    excludes=platform_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='backend_cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Required for stdin/stdout IPC
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='backend_cli',
)
