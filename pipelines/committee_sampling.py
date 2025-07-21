import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import torch
from scipy.stats import entropy

class CommitteeSamplingPipeline:
    def __init__(self, embedder, use_gpu_lightgbm=True, random_state=42):
        self.embedder = embedder
        self.random_state = random_state
        self.use_gpu_lightgbm = use_gpu_lightgbm
        self.committee = self._init_committee()
        
        self.labeled_embeddings = None
        self.labeled_labels = None
        self.unlabeled_embeddings = None
        self.unlabeled_texts = None
        self.unlabeled_indices = None
        
    def _init_committee(self):
        committee = []
        
        committee.append(LogisticRegression(random_state=self.random_state, max_iter=1000))
        committee.append(SVC(probability=True, random_state=self.random_state))
        
        # Always use CPU for LightGBM to avoid OpenCL issues
        committee.append(lgb.LGBMClassifier(
            device='cpu',
            random_state=self.random_state,
            verbose=-1
        ))
        
        committee.append(RandomForestClassifier(random_state=self.random_state, n_estimators=100))
        
        return committee
    
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
    
    def train_committee(self):
        print("Training committee of classifiers...")
        accuracies = {}
        
        for i, classifier in enumerate(self.committee):
            classifier_name = type(classifier).__name__
            print(f"Training {classifier_name}...")
            
            classifier.fit(self.labeled_embeddings, self.labeled_labels)
            
            predictions = classifier.predict(self.labeled_embeddings)
            accuracy = accuracy_score(self.labeled_labels, predictions)
            accuracies[classifier_name] = accuracy
            print(f"{classifier_name} training accuracy: {accuracy:.4f}")
        
        return accuracies
    
    def vote_entropy_sampling(self, n_samples: int = 10) -> Tuple[List[int], np.ndarray, List[str]]:
        if len(self.unlabeled_indices) == 0:
            return [], np.array([]), []
        
        all_probabilities = []
        for classifier in self.committee:
            probs = classifier.predict_proba(self.unlabeled_embeddings)
            all_probabilities.append(probs)
        
        all_probabilities = np.array(all_probabilities)
        
        vote_entropies = []
        for i in range(all_probabilities.shape[1]):
            sample_probs = all_probabilities[:, i, :]
            avg_probs = np.mean(sample_probs, axis=0)
            vote_entropy = entropy(avg_probs + 1e-8)
            vote_entropies.append(vote_entropy)
        
        vote_entropies = np.array(vote_entropies)
        
        n_samples = min(n_samples, len(self.unlabeled_indices))
        most_uncertain_idx = np.argsort(vote_entropies)[-n_samples:]
        
        selected_indices = [self.unlabeled_indices[i] for i in most_uncertain_idx]
        selected_entropies = vote_entropies[most_uncertain_idx]
        selected_texts = [self.unlabeled_texts[i] for i in most_uncertain_idx]
        
        return selected_indices, selected_entropies, selected_texts
    
    def disagreement_sampling(self, n_samples: int = 10) -> Tuple[List[int], np.ndarray, List[str]]:
        if len(self.unlabeled_indices) == 0:
            return [], np.array([]), []
        
        all_predictions = []
        for classifier in self.committee:
            preds = classifier.predict(self.unlabeled_embeddings)
            all_predictions.append(preds)
        
        all_predictions = np.array(all_predictions)
        
        disagreement_scores = []
        for i in range(all_predictions.shape[1]):
            sample_preds = all_predictions[:, i]
            unique_preds, counts = np.unique(sample_preds, return_counts=True)
            disagreement = 1.0 - (np.max(counts) / len(sample_preds))
            disagreement_scores.append(disagreement)
        
        disagreement_scores = np.array(disagreement_scores)
        
        n_samples = min(n_samples, len(self.unlabeled_indices))
        most_disagreed_idx = np.argsort(disagreement_scores)[-n_samples:]
        
        selected_indices = [self.unlabeled_indices[i] for i in most_disagreed_idx]
        selected_scores = disagreement_scores[most_disagreed_idx]
        selected_texts = [self.unlabeled_texts[i] for i in most_disagreed_idx]
        
        return selected_indices, selected_scores, selected_texts
    
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
    
    def get_committee_predictions(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        predictions = {}
        for classifier in self.committee:
            classifier_name = type(classifier).__name__
            preds = classifier.predict_proba(embeddings)
            predictions[classifier_name] = preds
        return predictions
    
    def run_active_learning_iteration(self, n_samples: int = 10, strategy: str = 'vote_entropy') -> Dict[str, Any]:
        accuracies = self.train_committee()
        
        if strategy == 'vote_entropy':
            selected_indices, scores, selected_texts = self.vote_entropy_sampling(n_samples)
        elif strategy == 'disagreement':
            selected_indices, scores, selected_texts = self.disagreement_sampling(n_samples)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        results = {
            'selected_indices': selected_indices,
            'scores': scores,
            'selected_texts': selected_texts,
            'pool_sizes': self.get_pool_sizes(),
            'committee_accuracies': accuracies,
            'strategy': strategy
        }
        
        print(f"Selected {len(selected_texts)} samples using {strategy} strategy")
        return results