import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, csv_path: str, text_column: str = 'text', label_column: str = 'label'):
        self.csv_path = csv_path
        self.text_column = text_column
        self.label_column = label_column
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.csv_path)
        return self.data
    
    def get_initial_split(self, initial_labeled_size: int = 100, random_state: int = 42) -> Tuple[List[str], List[int], List[str], List[int]]:
        if self.data is None:
            self.load_data()
            
        texts = self.data[self.text_column].tolist()
        labels = self.data[self.label_column].tolist()
        
        labeled_texts, unlabeled_texts, labeled_labels, unlabeled_labels = train_test_split(
            texts, labels, 
            train_size=initial_labeled_size, 
            random_state=random_state,
            stratify=labels
        )
        
        return labeled_texts, labeled_labels, unlabeled_texts, unlabeled_labels
    
    def get_full_dataset_for_active_learning(self, initial_labeled_indices: List[int]) -> Tuple[List[str], List[int], List[str], List[int]]:
        """Get full dataset with specified indices as initially labeled"""
        if self.data is None:
            self.load_data()
            
        texts = self.data[self.text_column].tolist()
        labels = self.data[self.label_column].tolist()
        
        labeled_texts = [texts[i] for i in initial_labeled_indices]
        labeled_labels = [labels[i] for i in initial_labeled_indices]
        
        unlabeled_indices = [i for i in range(len(texts)) if i not in initial_labeled_indices]
        unlabeled_texts = [texts[i] for i in unlabeled_indices]
        unlabeled_labels = [labels[i] for i in unlabeled_indices]  # Keep for simulation
        
        return labeled_texts, labeled_labels, unlabeled_texts, unlabeled_labels
    
    def get_text_and_labels(self) -> Tuple[List[str], List[int]]:
        if self.data is None:
            self.load_data()
        return self.data[self.text_column].tolist(), self.data[self.label_column].tolist()