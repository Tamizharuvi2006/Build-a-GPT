"""Document classes and loading utilities for RAG."""
import os
from dataclasses import dataclass
from typing import List

@dataclass
class Document:
    """Represents a text document for retrieval."""
    content: str
    metadata: dict
    doc_id: str

class DocumentLoader:
    """Utility to load and chunk text documents."""
    
    @staticmethod
    def load_text(path: str) -> Document:
        """Load a single text file."""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Document(content=content, metadata={"source": path}, doc_id=os.path.basename(path))
        
    @staticmethod
    def load_directory(dir_path: str) -> List[Document]:
        """Load all text files in a directory."""
        docs = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.txt'):
                    docs.append(DocumentLoader.load_text(os.path.join(root, file)))
        return docs
        
    @staticmethod
    def chunk_document(doc: Document, chunk_size: int = 512, overlap: int = 64) -> List[Document]:
        """Split a document into overlapping chunks."""
        chunks = []
        text = doc.content
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(Document(
                content=chunk_text,
                metadata={**doc.metadata, "chunk_idx": chunk_idx},
                doc_id=f"{doc.doc_id}_{chunk_idx}"
            ))
            start += (chunk_size - overlap)
            chunk_idx += 1
        return chunks
