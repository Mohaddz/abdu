import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional

class SBERTEmbedder:
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"SBERT model loaded on {self.device}")
        
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if self.model is None:
            self.load_model()
            
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def get_embedding_dim(self) -> int:
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()