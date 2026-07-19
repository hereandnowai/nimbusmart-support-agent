from dataclasses import dataclass

@dataclass
class Candidate:
    id: str
    text: str
    source: str
    section: str
    score: float = 0.0