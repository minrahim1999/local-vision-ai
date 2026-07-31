# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

# Hardcoded project root — this spec is always run from project root
project_root = Path.cwd()

block_cipher = None

a = Analysis(
    [str(project_root / 'app' / 'standalone.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / 'config'), 'config'),
        (str(project_root / 'services'), 'services'),
        (str(project_root / 'api'), 'api'),
        (str(project_root / 'assets' / 'icons'), 'assets/icons'),
    ],
    hiddenimports=[
        'flet',
        'flet_core',
        'flet_runtime',
        'requests',
        'PIL',
        'PIL.Image',
        'yaml',
        'fastapi',
        'pydantic',
        'uvicorn',
        'services.backends.factory',
        'services.backends.t2i_apple',
        'services.backends.t2i_diffusers',
        'services.backends.i2t_apple',
        'services.backends.i2t_transformers',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'diffusers',
        'transformers',
        'mlx',
        'mlx_vlm',
        'unsloth',
        'trl',
        'peft',
        'datasets',
        'sentencepiece',
    ],
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
    name='LocalVisionAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LocalVisionAI',
)

# macOS app bundle
app = BUNDLE(
    coll,
    name='LocalVisionAI.app',
    icon=str(project_root / 'assets' / 'icons' / 'icon.icns'),
    bundle_identifier='com.muhaimin.localvisionai',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'LSBackgroundOnly': 'False',
    },
)
