"""
Dense embedding-based retrieval using sentence transformers and FAISS.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import structlog

logger = structlog.get_logger()


class EmbeddingRetriever:
    """
    Dense retrieval using sentence embeddings and FAISS for efficient similarity search.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding retriever.
        
        Args:
            model_name: Name or path of the sentence transformer model
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner product for cosine similarity
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        
        logger.info("embedding_retriever_initialized", model_name=model_name)
    
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info("loading_embedding_model", model_name=self.model_name)
            self.model = SentenceTransformer(self.model_name)
            logger.info("embedding_model_loaded", 
                       model_name=self.model_name,
                       embedding_dim=self.model.get_sentence_embedding_dimension())
    
    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            
        Returns:
            Array of embeddings
        """
        if self.model is None:
            self.load_model()
        
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        if normalize:
            # L2 normalize for cosine similarity via inner product
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def index_documents(self, documents: List[Dict[str, Any]], text_field: str = "text"):
        """
        Index documents by computing embeddings and building FAISS index.
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing the text to embed
        """
        if not documents:
            logger.warning("no_documents_to_index")
            return
        
        self.documents = documents
        
        # Extract texts
        texts = [doc[text_field] for doc in documents]
        
        # Compute embeddings
        logger.info("computing_embeddings", num_documents=len(texts))
        self.embeddings = self.encode(texts, normalize=True)
        
        # Build FAISS index
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for normalized vectors
        self.index.add(self.embeddings.astype(np.float32))
        
        logger.info("embedding_index_built", 
                   num_documents=len(documents),
                   embedding_dim=embedding_dim)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using embedding similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents with similarity scores
        """
        if self.index is None or not self.documents:
            logger.warning("embedding_search_attempted_without_index")
            return []
        
        # Encode query
        query_embedding = self.encode([query], normalize=True)
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                result = self.documents[idx].copy()
                result["score"] = float(score)
                result["retrieval_method"] = "embedding"
                results.append(result)
        
        logger.debug("embedding_search_completed", 
                    query=query, 
                    num_results=len(results))
        
        return results
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.encode([query], normalize=True)[0]
    
    def save_index(self, filepath: str):
        """
        Save FAISS index to disk.
        
        Args:
            filepath: Path to save the index
        """
        if self.index is None:
            logger.warning("no_index_to_save")
            return
        
        faiss.write_index(self.index, filepath)
        logger.info("index_saved", filepath=filepath)
    
    def load_index(self, filepath: str):
        """
        Load FAISS index from disk.
        
        Args:
            filepath: Path to load the index from
        """
        self.index = faiss.read_index(filepath)
        logger.info("index_loaded", filepath=filepath)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"num_documents": 0, "indexed": False}
        
        return {
            "num_documents": len(self.documents),
            "indexed": True,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "model_name": self.model_name
        }


