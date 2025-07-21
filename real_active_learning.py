#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.embeddings import SBERTEmbedder
from pipelines.uncertainty_sampling import UncertaintySamplingPipeline
from pipelines.committee_sampling import CommitteeSamplingPipeline

class RealActivelearningDataLoader:
    def __init__(self, labeled_csv_path: str, unlabeled_csv_path: str):
        self.labeled_csv_path = labeled_csv_path
        self.unlabeled_csv_path = unlabeled_csv_path
        self.labeled_data = None
        self.unlabeled_data = None
        
    def load_data(self):
        """Load both labeled and unlabeled datasets"""
        print("Loading labeled dataset...")
        self.labeled_data = pd.read_csv(self.labeled_csv_path)
        print(f"Labeled data shape: {self.labeled_data.shape}")
        
        print("Loading unlabeled dataset...")
        self.unlabeled_data = pd.read_csv(self.unlabeled_csv_path)
        print(f"Unlabeled data shape: {self.unlabeled_data.shape}")
        
        # Show data info
        print(f"\nLabeled data columns: {list(self.labeled_data.columns)}")
        print(f"Unlabeled data columns: {list(self.unlabeled_data.columns)}")
        
        return self.labeled_data, self.unlabeled_data
    
    def prepare_for_active_learning(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Prepare data for active learning:
        - Extract labeled texts and labels
        - Extract unlabeled texts
        """
        if self.labeled_data is None or self.unlabeled_data is None:
            self.load_data()
        
        # Extract labeled data
        labeled_texts = self.labeled_data['text'].tolist()
        
        # Convert labels to binary (security = 1, non-security = 0)
        labels = []
        for label in self.labeled_data['label']:
            if label == 'security':
                labels.append(1)
            elif label == 'non-security':
                labels.append(0)
            else:
                # Handle any other label format
                labels.append(1 if 'security' in str(label).lower() else 0)
        
        # Extract unlabeled data (use 'content' column)
        unlabeled_texts = self.unlabeled_data['content'].tolist()
        
        print(f"\nPrepared for Active Learning:")
        print(f"├── Labeled texts: {len(labeled_texts)}")
        print(f"├── Label distribution: {np.bincount(labels)} (0=non-security, 1=security)")
        print(f"└── Unlabeled texts: {len(unlabeled_texts)}")
        
        return labeled_texts, labels, unlabeled_texts

def get_human_labels(selected_texts: List[str], selected_indices: List[int]) -> List[int]:
    """
    Function to get human labels for selected texts
    In real scenario, this would involve human annotation
    For now, we'll use a simple keyword-based simulation
    """
    print(f"\n" + "="*80)
    print("HUMAN LABELING SIMULATION")
    print("="*80)
    print("In a real scenario, you would manually label these texts.")
    print("For demonstration, using keyword-based simulation.")
    
    labels = []
    security_keywords = [
        'خصوصية', 'حماية', 'أمان', 'تجسس', 'اختراق', 'فيروس', 'برمجية خبيثة',
        'سرية', 'تشفير', 'قرصنة', 'احتيال', 'نصب', 'خداع', 'سرقة',
        'حساب', 'كلمة مرور', 'بيانات شخصية', 'معلومات حساسة'
    ]
    
    for i, text in enumerate(selected_texts):
        # Simulate human labeling based on security keywords
        is_security = any(keyword in text for keyword in security_keywords)
        label = 1 if is_security else 0
        
        print(f"\n{i+1}. Text: {text[:100]}...")
        print(f"   Simulated Label: {'Security' if label == 1 else 'Non-Security'}")
        
        labels.append(label)
    
    print(f"\nLabeling completed: {len(labels)} texts labeled")
    return labels

def run_uncertainty_sampling(labeled_texts: List[str], labeled_labels: List[int], 
                            unlabeled_texts: List[str], n_iterations: int = 3):
    """Run uncertainty sampling pipeline"""
    print("\n" + "="*80)
    print("UNCERTAINTY SAMPLING PIPELINE - REAL DATA")
    print("="*80)
    
    # Initialize embedder and pipeline
    embedder = SBERTEmbedder()
    pipeline = UncertaintySamplingPipeline(
        embedder=embedder,
        classifier_type='logistic',
        random_state=42
    )
    
    # Initialize pools
    pipeline.initialize_pools(
        labeled_texts.copy(),
        labeled_labels.copy(),
        unlabeled_texts.copy()
    )
    
    results_history = []
    
    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*60}")
        
        # Run active learning iteration
        results = pipeline.run_active_learning_iteration(
            n_samples=15,  # Select 15 most uncertain samples
            strategy='least_confident'
        )
        
        pool_sizes = results['pool_sizes']
        print(f"Current pool sizes: Labeled={pool_sizes['labeled']}, Unlabeled={pool_sizes['unlabeled']}")
        
        if len(results['selected_texts']) == 0:
            print("No more samples to select. Stopping.")
            break
            
        print(f"\nSelected {len(results['selected_texts'])} most uncertain samples:")
        for i, (text, uncertainty) in enumerate(zip(results['selected_texts'], results['uncertainties'])):
            print(f"{i+1}. Uncertainty: {uncertainty:.4f}")
            print(f"   Text: {text[:100]}...")
        
        # Get human labels (simulated)
        human_labels = get_human_labels(results['selected_texts'], results['selected_indices'])
        
        # Add newly labeled samples to the pipeline
        pipeline.add_labeled_samples(
            results['selected_texts'],
            human_labels,
            results['selected_indices']
        )
        
        results_history.append(results)
        
        print(f"\nIteration {iteration + 1} completed!")
        print(f"Total labeled samples: {pipeline.get_pool_sizes()['labeled']}")
    
    return pipeline, results_history

def run_committee_sampling(labeled_texts: List[str], labeled_labels: List[int], 
                          unlabeled_texts: List[str], n_iterations: int = 3):
    """Run committee sampling pipeline"""
    print("\n" + "="*80)
    print("COMMITTEE SAMPLING PIPELINE - REAL DATA")
    print("="*80)
    
    # Initialize embedder and pipeline
    embedder = SBERTEmbedder()
    pipeline = CommitteeSamplingPipeline(
        embedder=embedder,
        use_gpu_lightgbm=False,  # Use CPU to avoid OpenCL issues
        random_state=42
    )
    
    # Initialize pools
    pipeline.initialize_pools(
        labeled_texts.copy(),
        labeled_labels.copy(),
        unlabeled_texts.copy()
    )
    
    results_history = []
    
    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*60}")
        
        # Run active learning iteration
        results = pipeline.run_active_learning_iteration(
            n_samples=15,  # Select 15 most disagreed samples
            strategy='vote_entropy'
        )
        
        pool_sizes = results['pool_sizes']
        print(f"Current pool sizes: Labeled={pool_sizes['labeled']}, Unlabeled={pool_sizes['unlabeled']}")
        
        # Show committee accuracies
        print("\nCommittee Performance:")
        for classifier_name, accuracy in results['committee_accuracies'].items():
            print(f"├── {classifier_name}: {accuracy:.4f}")
        
        if len(results['selected_texts']) == 0:
            print("No more samples to select. Stopping.")
            break
            
        print(f"\nSelected {len(results['selected_texts'])} most disagreed samples:")
        for i, (text, score) in enumerate(zip(results['selected_texts'], results['scores'])):
            print(f"{i+1}. Vote Entropy: {score:.4f}")
            print(f"   Text: {text[:100]}...")
        
        # Get human labels (simulated)
        human_labels = get_human_labels(results['selected_texts'], results['selected_indices'])
        
        # Add newly labeled samples to the pipeline
        pipeline.add_labeled_samples(
            results['selected_texts'],
            human_labels,
            results['selected_indices']
        )
        
        results_history.append(results)
        
        print(f"\nIteration {iteration + 1} completed!")
        print(f"Total labeled samples: {pipeline.get_pool_sizes()['labeled']}")
    
    return pipeline, results_history

def main():
    """Main execution function"""
    print("="*80)
    print("REAL ACTIVE LEARNING WITH YOUR DATA")
    print("="*80)
    print("Using:")
    print("├── Labeled data: balanced_label_data_300_ROWS.csv")
    print("└── Unlabeled data: preprocessed_unlabeled_data.csv")
    
    # Initialize data loader
    data_loader = RealActivelearningDataLoader(
        labeled_csv_path="data/balanced_label_data_300_ROWS.csv",
        unlabeled_csv_path="data/preprocessed_unlabeled_data.csv"
    )
    
    # Load and prepare data
    labeled_texts, labeled_labels, unlabeled_texts = data_loader.prepare_for_active_learning()
    
    try:
        # Run Uncertainty Sampling Pipeline
        print(f"\n🎯 Starting Uncertainty Sampling...")
        uncertainty_pipeline, uncertainty_results = run_uncertainty_sampling(
            labeled_texts, labeled_labels, unlabeled_texts, n_iterations=2
        )
        
        # Run Committee Sampling Pipeline (fresh start)
        print(f"\n🎯 Starting Committee Sampling...")
        committee_pipeline, committee_results = run_committee_sampling(
            labeled_texts, labeled_labels, unlabeled_texts, n_iterations=2
        )
        
        # Final Summary
        print("\n" + "="*80)
        print("ACTIVE LEARNING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        uncertainty_final = uncertainty_pipeline.get_pool_sizes()
        committee_final = committee_pipeline.get_pool_sizes()
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"")
        print(f"🔍 Uncertainty Sampling:")
        print(f"├── Started with: {len(labeled_texts)} labeled samples")
        print(f"├── Final labeled pool: {uncertainty_final['labeled']} samples")
        print(f"└── Remaining unlabeled: {uncertainty_final['unlabeled']} samples")
        
        print(f"")
        print(f"👥 Committee Sampling:")
        print(f"├── Started with: {len(labeled_texts)} labeled samples") 
        print(f"├── Final labeled pool: {committee_final['labeled']} samples")
        print(f"└── Remaining unlabeled: {committee_final['unlabeled']} samples")
        
        print(f"\n✅ Both pipelines successfully used your real data!")
        print(f"✅ Labeled dataset: {len(labeled_texts)} Arabic texts")
        print(f"✅ Unlabeled dataset: {len(unlabeled_texts)} Arabic texts")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()