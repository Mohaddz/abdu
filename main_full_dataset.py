#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.embeddings import SBERTEmbedder
from pipelines.uncertainty_sampling import UncertaintySamplingPipeline
from pipelines.committee_sampling import CommitteeSamplingPipeline

def run_active_learning_on_full_dataset(
    csv_path: str, 
    initial_labeled_indices: List[int] = None,
    n_iterations: int = 3,
    samples_per_iteration: int = 20
):
    """
    Run active learning on full dataset without splitting
    
    Args:
        csv_path: Path to your CSV file with 'text' and 'label' columns
        initial_labeled_indices: List of indices to start with as labeled (if None, uses first 50)
        n_iterations: Number of active learning iterations to run
        samples_per_iteration: Number of samples to select per iteration
    """
    
    print("=" * 80)
    print("ARABIC TEXT CLASSIFICATION - ACTIVE LEARNING ON FULL DATASET")
    print("=" * 80)
    
    # Load data
    data_loader = DataLoader(csv_path)
    df = data_loader.load_data()
    
    print(f"Dataset loaded: {len(df)} total samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Set initial labeled indices if not provided
    if initial_labeled_indices is None:
        # Use first 50 samples as initially labeled
        initial_labeled_indices = list(range(50))
    
    print(f"Starting with {len(initial_labeled_indices)} labeled samples")
    
    # Get data splits
    labeled_texts, labeled_labels, unlabeled_texts, unlabeled_labels = data_loader.get_full_dataset_for_active_learning(
        initial_labeled_indices
    )
    
    # Initialize embedder
    embedder = SBERTEmbedder()
    
    # Run Uncertainty Sampling
    print("\n" + "=" * 60)
    print("UNCERTAINTY SAMPLING PIPELINE")
    print("=" * 60)
    
    uncertainty_pipeline = UncertaintySamplingPipeline(
        embedder=embedder, 
        classifier_type='logistic',
        random_state=42
    )
    
    uncertainty_pipeline.initialize_pools(
        labeled_texts.copy(), 
        labeled_labels.copy(), 
        unlabeled_texts.copy()
    )
    
    uncertainty_results = []
    for iteration in range(n_iterations):
        print(f"\n--- Uncertainty Sampling Iteration {iteration + 1} ---")
        
        results = uncertainty_pipeline.run_active_learning_iteration(
            n_samples=samples_per_iteration, 
            strategy='least_confident'
        )
        
        print(f"Pool sizes: Labeled={results['pool_sizes']['labeled']}, Unlabeled={results['pool_sizes']['unlabeled']}")
        print(f"Top 3 most uncertain samples:")
        
        for i, (text, uncertainty) in enumerate(zip(results['selected_texts'][-3:], results['uncertainties'][-3:])):
            print(f"{i+1}. Uncertainty: {uncertainty:.4f}")
            print(f"   Text: {text[:100]}...")
        
        # Simulate human labeling (in real scenario, you would manually label these)
        simulated_labels = []
        for text in results['selected_texts']:
            # Simple simulation: if contains security keywords, label as 1
            if any(keyword in text for keyword in ['أمان', 'حماية', 'تسريب', 'اختراق', 'فيروس', 'برمجيات خبيثة', 'قرصنة']):
                simulated_labels.append(1)
            else:
                simulated_labels.append(0)
        
        uncertainty_pipeline.add_labeled_samples(
            results['selected_texts'], 
            simulated_labels, 
            results['selected_indices']
        )
        
        uncertainty_results.append(results)
    
    # Run Committee Sampling (fresh start)
    print("\n" + "=" * 60)
    print("COMMITTEE SAMPLING PIPELINE")
    print("=" * 60)
    
    committee_pipeline = CommitteeSamplingPipeline(
        embedder=embedder,
        use_gpu_lightgbm=False,  # Disabled to avoid OpenCL issues
        random_state=42
    )
    
    committee_pipeline.initialize_pools(
        labeled_texts.copy(), 
        labeled_labels.copy(), 
        unlabeled_texts.copy()
    )
    
    committee_results = []
    for iteration in range(n_iterations):
        print(f"\n--- Committee Sampling Iteration {iteration + 1} ---")
        
        results = committee_pipeline.run_active_learning_iteration(
            n_samples=samples_per_iteration,
            strategy='vote_entropy'
        )
        
        print(f"Pool sizes: Labeled={results['pool_sizes']['labeled']}, Unlabeled={results['pool_sizes']['unlabeled']}")
        print("Committee accuracies:", {k: f"{v:.4f}" for k, v in results['committee_accuracies'].items()})
        print(f"Top 3 most disagreed samples:")
        
        for i, (text, score) in enumerate(zip(results['selected_texts'][-3:], results['scores'][-3:])):
            print(f"{i+1}. Vote Entropy: {score:.4f}")
            print(f"   Text: {text[:100]}...")
        
        # Simulate human labeling
        simulated_labels = []
        for text in results['selected_texts']:
            if any(keyword in text for keyword in ['أمان', 'حماية', 'تسريب', 'اختراق', 'فيروس', 'برمجيات خبيثة', 'قرصنة']):
                simulated_labels.append(1)
            else:
                simulated_labels.append(0)
        
        committee_pipeline.add_labeled_samples(
            results['selected_texts'], 
            simulated_labels, 
            results['selected_indices']
        )
        
        committee_results.append(results)
    
    # Final Summary
    print("\n" + "=" * 80)
    print("ACTIVE LEARNING COMPLETED")
    print("=" * 80)
    
    final_uncertainty_size = uncertainty_pipeline.get_pool_sizes()
    final_committee_size = committee_pipeline.get_pool_sizes()
    
    print(f"\nUncertainty Sampling Final State:")
    print(f"  Labeled pool: {final_uncertainty_size['labeled']} samples")
    print(f"  Unlabeled pool: {final_uncertainty_size['unlabeled']} samples")
    
    print(f"\nCommittee Sampling Final State:")
    print(f"  Labeled pool: {final_committee_size['labeled']} samples")
    print(f"  Unlabeled pool: {final_committee_size['unlabeled']} samples")
    
    print(f"\nTotal samples processed from dataset: {len(df)}")
    print(f"Initial labeled: {len(initial_labeled_indices)}")
    print(f"Samples added per pipeline: {n_iterations * samples_per_iteration}")
    
    return uncertainty_results, committee_results

def main():
    """Main execution function"""
    
    # Example usage with your CSV file
    csv_path = "data/sample_arabic_data.csv"
    
    # Create sample data if it doesn't exist
    if not os.path.exists(csv_path):
        print("Creating sample data for demonstration...")
        from main import create_sample_data
        create_sample_data(csv_path)
    
    # Define which samples to start with as labeled (indices)
    # You can customize this list based on your needs
    initial_labeled_indices = list(range(0, 100, 2))  # Every other sample from first 100
    
    # Run active learning
    uncertainty_results, committee_results = run_active_learning_on_full_dataset(
        csv_path=csv_path,
        initial_labeled_indices=initial_labeled_indices,
        n_iterations=2,  # Number of AL iterations
        samples_per_iteration=15  # Samples to select per iteration
    )
    
    print("\n" + "="*80)
    print("READY FOR YOUR DATA!")
    print("="*80)
    print("To use with your own CSV file:")
    print("1. Update csv_path to your file")
    print("2. Customize initial_labeled_indices")
    print("3. Adjust n_iterations and samples_per_iteration")
    print("4. Replace simulated labeling with real human annotation")

if __name__ == "__main__":
    main()