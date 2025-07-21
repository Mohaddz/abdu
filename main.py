#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.embeddings import SBERTEmbedder
from pipelines.uncertainty_sampling import UncertaintySamplingPipeline
from pipelines.committee_sampling import CommitteeSamplingPipeline

def create_sample_data(output_path: str = "data/sample_arabic_data.csv", n_samples: int = 1000):
    """Create sample Arabic text data for testing"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    security_texts = [
        "تسريب البيانات الشخصية للمستخدمين",
        "هجمات القرصنة الإلكترونية على البنوك",
        "اختراق أنظمة الحماية الرقمية",
        "برمجيات خبيثة تهدد الشبكات",
        "فيروسات الحاسوب والبرمجيات الضارة",
        "سرقة كلمات المرور والحسابات",
        "أمان المعلومات والحماية الرقمية",
        "تشفير البيانات الحساسة",
        "حماية الخصوصية على الإنترنت",
        "أمان الشبكات والخوادم"
    ]
    
    non_security_texts = [
        "الطقس اليوم مشمس وجميل",
        "وصفة الطبخ التقليدية العربية",
        "أخبار الرياضة والمباريات",
        "السياحة والسفر إلى البلدان العربية",
        "التعليم والثقافة في المجتمع",
        "الفنون والموسيقى العربية",
        "الأدب والشعر الكلاسيكي",
        "التجارة والاقتصاد المحلي",
        "الصحة والطب التقليدي",
        "التكنولوجيا والابتكار"
    ]
    
    data = []
    for i in range(n_samples):
        if i % 2 == 0:
            text = np.random.choice(security_texts)
            label = 1
        else:
            text = np.random.choice(non_security_texts)
            label = 0
        
        data.append({
            'text': f"{text} - عينة رقم {i+1}",
            'label': label
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Sample data created at: {output_path}")
    return output_path

def run_uncertainty_sampling_demo(data_path: str):
    """Run uncertainty sampling pipeline demonstration"""
    print("\n" + "="*60)
    print("UNCERTAINTY SAMPLING PIPELINE DEMO")
    print("="*60)
    
    data_loader = DataLoader(data_path)
    embedder = SBERTEmbedder()
    
    labeled_texts, labeled_labels, unlabeled_texts, unlabeled_labels = data_loader.get_initial_split(
        initial_labeled_size=50
    )
    
    pipeline = UncertaintySamplingPipeline(embedder, classifier_type='logistic')
    pipeline.initialize_pools(labeled_texts, labeled_labels, unlabeled_texts)
    
    for iteration in range(2):
        print(f"\n--- Iteration {iteration + 1} ---")
        results = pipeline.run_active_learning_iteration(n_samples=10, strategy='least_confident')
        
        print(f"Pool sizes: {results['pool_sizes']}")
        print(f"Selected {len(results['selected_texts'])} samples")
        print("Top 3 most uncertain samples:")
        for i, (text, uncertainty) in enumerate(zip(results['selected_texts'][-3:], results['uncertainties'][-3:])):
            print(f"{i+1}. Uncertainty: {uncertainty:.4f} - {text[:80]}...")
        
        simulated_labels = [1 if 'أمان' in text or 'حماية' in text or 'تسريب' in text else 0 
                           for text in results['selected_texts']]
        pipeline.add_labeled_samples(results['selected_texts'], simulated_labels, results['selected_indices'])
    
    print("\nUncertainty Sampling Pipeline completed successfully!")
    return pipeline

def run_committee_sampling_demo(data_path: str):
    """Run committee sampling pipeline demonstration"""
    print("\n" + "="*60)
    print("COMMITTEE SAMPLING PIPELINE DEMO")
    print("="*60)
    
    data_loader = DataLoader(data_path)
    embedder = SBERTEmbedder()
    
    labeled_texts, labeled_labels, unlabeled_texts, unlabeled_labels = data_loader.get_initial_split(
        initial_labeled_size=50
    )
    
    pipeline = CommitteeSamplingPipeline(embedder, use_gpu_lightgbm=True)
    pipeline.initialize_pools(labeled_texts, labeled_labels, unlabeled_texts)
    
    for iteration in range(2):
        print(f"\n--- Iteration {iteration + 1} ---")
        results = pipeline.run_active_learning_iteration(n_samples=10, strategy='vote_entropy')
        
        print(f"Pool sizes: {results['pool_sizes']}")
        print("Committee accuracies:", {k: f"{v:.4f}" for k, v in results['committee_accuracies'].items()})
        print(f"Selected {len(results['selected_texts'])} samples")
        print("Top 3 most disagreed samples:")
        for i, (text, score) in enumerate(zip(results['selected_texts'][-3:], results['scores'][-3:])):
            print(f"{i+1}. Vote Entropy: {score:.4f} - {text[:80]}...")
        
        simulated_labels = [1 if 'أمان' in text or 'حماية' in text or 'تسريب' in text else 0 
                           for text in results['selected_texts']]
        pipeline.add_labeled_samples(results['selected_texts'], simulated_labels, results['selected_indices'])
    
    print("\nCommittee Sampling Pipeline completed successfully!")
    return pipeline

def main():
    """Main execution function"""
    print("Arabic Text Classification - Active Learning Pipelines")
    print("====================================================")
    
    data_path = "data/sample_arabic_data.csv"
    
    if not os.path.exists(data_path):
        print("Creating sample Arabic data...")
        create_sample_data(data_path)
    
    try:
        uncertainty_pipeline = run_uncertainty_sampling_demo(data_path)
        committee_pipeline = run_committee_sampling_demo(data_path)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Both active learning pipelines have been demonstrated.")
        print("You can now use your own Arabic CSV data by modifying the data_path variable.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()