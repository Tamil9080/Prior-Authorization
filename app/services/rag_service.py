import os
import pypdf
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "cms_mcd"

class PolicyRAGEngine:
    def __init__(self):
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.is_indexed = False
        
    def initialize(self):
        """
        Load PDFs from raw directory, split into text chunks, and fit the TF-IDF vectorizer.
        """
        try:
            raw_documents = []
            
            # Read all PDF and TXT files in PDF_DIR
            if PDF_DIR.exists():
                for path in PDF_DIR.iterdir():
                    if path.suffix.lower() == ".pdf":
                        try:
                            reader = pypdf.PdfReader(path)
                            source_name = path.stem.replace("_", " ").title()
                            for i, page in enumerate(reader.pages):
                                text = page.extract_text() or ""
                                raw_documents.append((text, f"{source_name} (Page {i+1})"))
                        except Exception as e:
                            print(f"Error reading PDF {path.name} in RAG: {e}")
                    elif path.suffix.lower() == ".txt":
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                text = f.read()
                            source_name = path.stem.replace("_", " ").title()
                            raw_documents.append((text, f"{source_name} Guide"))
                        except Exception as e:
                            print(f"Error reading TXT {path.name} in RAG: {e}")
            
            # Chunking documents
            self.chunks = []
            for doc_text, source in raw_documents:
                # Chunk by paragraphs/double newlines, or split into fixed-size segments
                paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
                for p in paragraphs:
                    if len(p) < 60:  # Skip trivial headers
                        continue
                    self.chunks.append({
                        "text": p,
                        "source": source
                    })
                    
            if self.chunks:
                corpus = [c["text"] for c in self.chunks]
                self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
                self.is_indexed = True
                print(f"RAG layer initialized and indexed {len(self.chunks)} text chunks.")
            else:
                print("Warning: RAG index is empty. No text chunks found in PDFs.")
                
        except Exception as e:
            print(f"Failed to initialize RAG index: {e}")
            
    def retrieve(self, query, top_k=2):
        """
        Perform semantic TF-IDF query against the indexed policy chunks.
        Returns a list of matching chunk dicts.
        """
        if not self.is_indexed or not query:
            return []
            
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top k indices
            top_indices = similarities.argsort()[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score > 0.05:  # Relevance threshold
                    chunk = self.chunks[idx].copy()
                    chunk["relevance_score"] = score
                    results.append(chunk)
            return results
        except Exception as e:
            print(f"Error during RAG retrieval: {e}")
            return []

# Singleton instance
rag_engine = PolicyRAGEngine()
