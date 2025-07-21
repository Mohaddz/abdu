import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import os
import hashlib

class SBERTEmbedder:
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 
                 cache_dir: str = 'embeddings_cache'):
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_model(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"SBERT model loaded on {self.device}")
        
    def _get_cache_filename(self, texts: List[str]) -> str:
        """Generate cache filename based on texts hash"""
        text_hash = hashlib.md5(''.join(texts).encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_{text_hash}.npy")
    
    def _load_cached_embeddings(self, cache_file: str) -> Optional[np.ndarray]:
        """Load embeddings from cache if exists"""
        if os.path.exists(cache_file):
            try:
                embeddings = np.load(cache_file)
                print(f"✅ Loaded cached embeddings from {cache_file}")
                return embeddings
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}")
        return None
    
    def _save_embeddings_to_cache(self, embeddings: np.ndarray, cache_file: str):
        """Save embeddings to cache"""
        try:
            np.save(cache_file, embeddings)
            print(f"💾 Saved embeddings to cache: {cache_file}")
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")
        
    def encode_texts(self, texts: List[str], batch_size: int = 512, use_cache: bool = True) -> np.ndarray:
        if self.model is None:
            self.load_model()
        
        # Check cache first
        if use_cache:
            cache_file = self._get_cache_filename(texts)
            cached_embeddings = self._load_cached_embeddings(cache_file)
            if cached_embeddings is not None:
                return cached_embeddings
        
        print(f"🚀 Generating embeddings for {len(texts)} texts with batch size {batch_size}...")
        
        # Generate embeddings with optimized settings
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            num_workers=4  # Multi-threading for data loading
        )
        
        # Save to cache
        if use_cache:
            self._save_embeddings_to_cache(embeddings, cache_file)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()