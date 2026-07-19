from dataclasses import dataclass
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2")]

_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "],
)

@dataclass
class Chunk:
    text: str
    source: str
    section: str
    chunk_index: int

def chunk_document(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    header_splitter = MarkdownHeaderTextSplitter(HEADERS_TO_SPLIT_ON, strip_headers=False)
    sections = header_splitter.split_text(raw)

    chunks: list[Chunk] = []
    for section in sections:
        section_title = section.metadata.get("h2") or path.stem
        for piece in _char_splitter.split_text(section.page_content):
            chunks.append(
                Chunk(text=piece, source=path.name, section=section_title, chunk_index=len(chunks)))
    return chunks

def chunk_all(documents_dir: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(documents_dir.glob("*.md")):
        all_chunks.extend(chunk_document(path))
    return all_chunks