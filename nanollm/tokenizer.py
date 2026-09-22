from typing import List

class ByteTokenizer:
    def __init__(self):
        self.pad_id = 0
        self.q_marker_id = 1

    @property
    def vocab_size(self) -> int:
        return 258

    def encode(self, text: str) -> List[int]:
        return [b + 2 for b in text.encode("utf-8")]

    def decode(self, token_ids: List[int]) -> str:
        valid_bytes = [b - 2 for b in token_ids if b >= 2]
        return bytes(valid_bytes).decode("utf-8", errors="ignore")

class SubwordTokenizer:
    def __init__(self, name: str = "answerdotai/ModernBERT-base"):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(name)
        self.pad_id = self._tok.pad_token_id if self._tok.pad_token_id is not None else 50283
        self.q_marker_id = self._tok.sep_token_id if self._tok.sep_token_id is not None else 50282

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=True)
