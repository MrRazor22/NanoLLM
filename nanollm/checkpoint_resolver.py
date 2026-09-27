from pathlib import Path
import re
from typing import Optional, Tuple

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

__all__ = ['CHECKPOINTS_DIR', 'get_latest_checkpoint', 'get_next_checkpoint_path']
