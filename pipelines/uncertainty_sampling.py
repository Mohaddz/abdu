import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import torch

class UncertaintySamplingPipeline:
    def __init__(self, embedder, classifier_type='logistic', random_state=42):
        self.embedder = embedder
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.classifier = self._init_classifier()
        
        self.labeled_embeddings = None
        self.labeled_labels = None
        self.unlabeled_embeddings = None
        self.unlabeled_texts = None
        self.unlabeled_indices = None
        
    def _init_classifier(self):
        if self.classifier_type == 'logistic':
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif self.classifier_type == 'svm':
            return SVC(probability=True, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
    
    def initialize_pools(self, labeled_texts: List[str], labeled_labels: List[int], 
                        unlabeled_texts: List[str]):
        print("Generating embeddings for labeled data...")
        self.labeled_embeddings = self.embedder.encode_texts(labeled_texts)
        self.labeled_labels = np.array(labeled_labels)
        
        print("Generating embeddings for unlabeled data...")
        self.unlabeled_embeddings = self.embedder.encode_texts(unlabeled_texts)
        self.unlabeled_texts = unlabeled_texts
        self.unlabeled_indices = list(range(len(unlabeled_texts)))
        
        print(f"Initialized with {len(labeled_texts)} labeled and {len(unlabeled_texts)} unlabeled samples")
    
    def train_classifier(self):
        print(f"Training {self.classifier_type} classifier...")
        self.classifier.fit(self.labeled_embeddings, self.labeled_labels)
        
        predictions = self.classifier.predict(self.labeled_embeddings)
        accuracy = accuracy_score(self.labeled_labels, predictions)
        print(f"Training accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def uncertainty_sampling(self, n_samples: int = 10, strategy: str = 'least_confident') -> Tuple[List[int], np.ndarray, List[str]]:
        if len(self.unlabeled_indices) == 0:
            return [], np.array([]), []
            
        probabilities = self.classifier.predict_proba(self.unlabeled_embeddings)
        
        if strategy == 'least_confident':
            uncertainties = 1 - np.max(probabilities, axis=1)
        elif strategy == 'margin':
            sorted_probs = np.sort(probabilities, axis=1)
            uncertainties = sorted_probs[:, -1] - sorted_probs[:, -2]
            uncertainties = 1 - uncertainties
        elif strategy == 'entropy':
            uncertainties = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        n_samples = min(n_samples, len(self.unlabeled_indices))
        most_uncertain_idx = np.argsort(uncertainties)[-n_samples:]
        
        selected_indices = [self.unlabeled_indices[i] for i in most_uncertain_idx]
        selected_uncertainties = uncertainties[most_uncertain_idx]
        selected_texts = [self.unlabeled_texts[i] for i in most_uncertain_idx]
        
        return selected_indices, selected_uncertainties, selected_texts
    
    def add_labeled_samples(self, new_texts: List[str], new_labels: List[int], selected_indices: List[int]):
        new_embeddings = self.embedder.encode_texts(new_texts)
        
        self.labeled_embeddings = np.vstack([self.labeled_embeddings, new_embeddings])
        self.labeled_labels = np.concatenate([self.labeled_labels, new_labels])
        
        indices_to_remove = set(selected_indices)
        remaining_indices = []
        remaining_texts = []
        remaining_embeddings = []
        
        for i, (idx, text, emb) in enumerate(zip(self.unlabeled_indices, self.unlabeled_texts, self.unlabeled_embeddings)):
            if idx not in indices_to_remove:
                remaining_indices.append(idx)
                remaining_texts.append(text)
                remaining_embeddings.append(emb)
        
        self.unlabeled_indices = remaining_indices
        self.unlabeled_texts = remaining_texts
        self.unlabeled_embeddings = np.array(remaining_embeddings) if remaining_embeddings else np.empty((0, self.labeled_embeddings.shape[1]))
        
        print(f"Added {len(new_texts)} samples to labeled pool. Remaining unlabeled: {len(self.unlabeled_texts)}")
    
    def get_pool_sizes(self) -> Dict[str, int]:
        return {
            'labeled': len(self.labeled_labels),
            'unlabeled': len(self.unlabeled_texts)
        }
    
    def run_active_learning_iteration(self, n_samples: int = 10, strategy: str = 'least_confident') -> Dict[str, Any]:
        self.train_classifier()
        
        selected_indices, uncertainties, selected_texts = self.uncertainty_sampling(n_samples, strategy)
        
        results = {
            'selected_indices': selected_indices,
            'uncertainties': uncertainties,
            'selected_texts': selected_texts,
            'pool_sizes': self.get_pool_sizes(),
            'strategy': strategy
        }
        
        print(f"Selected {len(selected_texts)} samples using {strategy} strategy")
        return results