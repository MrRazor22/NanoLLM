import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / 'checkpoints'

def get_latest_checkpoint(prefix: str = 'checkpoint_champion') -> Tuple[Optional[Path], int]:
    if not CHECKPOINTS_DIR.exists():
        return None, 0
    max_ver = -1
    best_file = None
    pattern = re.compile(rf'^{re.escape(prefix)}(?:_v(\d+))?\.pt$')
    for f in CHECKPOINTS_DIR.glob(f'{prefix}*.pt'):
        m = pattern.match(f.name)
        if m:
            ver = int(m.group(1)) if m.group(1) is not None else 1
            if ver > max_ver:
                max_ver = ver
                best_file = f
    return best_file, max(0, max_ver)

def get_next_checkpoint_path(prefix: str = 'checkpoint_champion') -> Path:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    _, latest_ver = get_latest_checkpoint(prefix)
    next_ver = latest_ver + 1
    return CHECKPOINTS_DIR / f'{prefix}_v{next_ver}.pt'

def list_all_checkpoints() -> List[Dict[str, Any]]:
    if not CHECKPOINTS_DIR.exists():
        return []
    manifest_path = CHECKPOINTS_DIR / 'manifest.json'
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except Exception:
            pass
    latest_ckpt, _ = get_latest_checkpoint()
    pattern = re.compile(r'^checkpoint_champion(?:_v(\d+))?\.pt$')
    results = []
    for f in CHECKPOINTS_DIR.glob('*.pt'):
        m = pattern.match(f.name)
        ver = int(m.group(1)) if (m and m.group(1) is not None) else (1 if m else 0)
        is_latest = (latest_ckpt is not None and f.resolve() == latest_ckpt.resolve())
        meta = manifest.get(f.name, {})
        stat = f.stat()
        results.append({
            'name': f.name,
            'path': f,
            'version': ver,
            'size_mb': round(stat.st_size / (1024 * 1024), 1),
            'modified': stat.st_mtime,
            'is_latest': is_latest,
            'train_loss': meta.get('train_loss'),
            'val_loss': meta.get('val_loss'),
            'step': meta.get('step'),
            'epoch': meta.get('epoch'),
            'saved_at': meta.get('saved_at'),
        })
    results.sort(key=lambda x: (x['version'] > 0, x['version'], x['name']), reverse=True)
    return results

__all__ = ['CHECKPOINTS_DIR', 'get_latest_checkpoint', 'get_next_checkpoint_path', 'list_all_checkpoints']

