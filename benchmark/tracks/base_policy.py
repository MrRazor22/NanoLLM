from typing import Any, Dict, List, Optional, Protocol

class ITrackPolicy(Protocol):
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]: ...
