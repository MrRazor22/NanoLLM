from typing import Any, Dict, List, Optional, Protocol

class ISuitePolicy(Protocol):
    """The bedrock contract for benchmark suite data providers."""
    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]: ...

# Backward compatibility alias
ITrackPolicy = ISuitePolicy

__all__ = ["ISuitePolicy", "ITrackPolicy"]
