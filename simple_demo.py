#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import List, Tuple
import os

def create_sample_data():
    """Create sample Arabic text data"""
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
    for i in range(500):
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
    
    return pd.DataFrame(data)

def uncertainty_sampling_demo():
    """Demonstrate uncertainty sampling with TF-IDF and simple classifiers"""
    print("=" * 60)
    print("SIMPLE UNCERTAINTY SAMPLING DEMO")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    print(f"Created {len(df)} samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Split into labeled and unlabeled
    labeled_data, unlabeled_data = train_test_split(
        df, train_size=50, random_state=42, stratify=df['label']
    )
    
    print(f"\nInitial pools:")
    print(f"Labeled: {len(labeled_data)} samples")
    print(f"Unlabeled: {len(unlabeled_data)} samples")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    
    # Fit on labeled data and transform
    labeled_vectors = vectorizer.fit_transform(labeled_data['text'])
    unlabeled_vectors = vectorizer.transform(unlabeled_data['text'])
    
    # Initialize classifier
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    
    # Active learning iterations
    for iteration in range(2):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Train classifier
        classifier.fit(labeled_vectors, labeled_data['label'])
        
        # Get training accuracy
        train_pred = classifier.predict(labeled_vectors)
        train_acc = accuracy_score(labeled_data['label'], train_pred)
        print(f"Training accuracy: {train_acc:.4f}")
        
        # Get predictions and uncertainties on unlabeled data
        probs = classifier.predict_proba(unlabeled_vectors)
        uncertainties = 1 - np.max(probs, axis=1)
        
        # Select most uncertain samples
        n_select = 10
        most_uncertain_idx = np.argsort(uncertainties)[-n_select:]
        
        # Get selected samples
        selected_samples = unlabeled_data.iloc[most_uncertain_idx].copy()
        selected_uncertainties = uncertainties[most_uncertain_idx]
        
        print(f"Selected {len(selected_samples)} most uncertain samples:")
        for i, (idx, row) in enumerate(selected_samples.iterrows()):
            uncertainty = selected_uncertainties[i]
            print(f"{i+1}. Uncertainty: {uncertainty:.4f} - {row['text'][:80]}...")
        
        # Simulate labeling (use true labels)
        new_labels = selected_samples['label'].tolist()
        
        # Add to labeled data
        labeled_data = pd.concat([labeled_data, selected_samples], ignore_index=True)
        
        # Remove from unlabeled data
        unlabeled_data = unlabeled_data.drop(selected_samples.index).reset_index(drop=True)
        
        # Re-vectorize
        labeled_vectors = vectorizer.fit_transform(labeled_data['text'])
        if len(unlabeled_data) > 0:
            unlabeled_vectors = vectorizer.transform(unlabeled_data['text'])
        
        print(f"Updated pools: Labeled={len(labeled_data)}, Unlabeled={len(unlabeled_data)}")
    
    print("\nUncertainty sampling demo completed!")
    return labeled_data, classifier

def committee_demo():
    """Demonstrate committee-based sampling"""
    print("\n" + "=" * 60)
    print("SIMPLE COMMITTEE SAMPLING DEMO")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    
    # Split into labeled and unlabeled
    labeled_data, unlabeled_data = train_test_split(
        df, train_size=50, random_state=42, stratify=df['label']
    )
    
    print(f"Initial pools:")
    print(f"Labeled: {len(labeled_data)} samples")
    print(f"Unlabeled: {len(unlabeled_data)} samples")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    
    # Initialize committee
    committee = [
        LogisticRegression(random_state=42, max_iter=1000),
        SVC(probability=True, random_state=42),
        RandomForestClassifier(random_state=42, n_estimators=50)
    ]
    
    # Active learning iterations
    for iteration in range(2):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Fit vectorizer and transform data
        labeled_vectors = vectorizer.fit_transform(labeled_data['text'])
        unlabeled_vectors = vectorizer.transform(unlabeled_data['text'])
        
        # Train committee
        committee_accuracies = {}
        all_probs = []
        
        for classifier in committee:
            classifier_name = type(classifier).__name__
            classifier.fit(labeled_vectors, labeled_data['label'])
            
            # Get training accuracy
            train_pred = classifier.predict(labeled_vectors)
            train_acc = accuracy_score(labeled_data['label'], train_pred)
            committee_accuracies[classifier_name] = train_acc
            
            # Get probabilities on unlabeled data
            probs = classifier.predict_proba(unlabeled_vectors)
            all_probs.append(probs)
        
        print("Committee accuracies:")
        for name, acc in committee_accuracies.items():
            print(f"  {name}: {acc:.4f}")
        
        # Calculate vote entropy
        all_probs = np.array(all_probs)  # (n_classifiers, n_samples, n_classes)
        avg_probs = np.mean(all_probs, axis=0)  # (n_samples, n_classes)
        vote_entropies = -np.sum(avg_probs * np.log(avg_probs + 1e-8), axis=1)
        
        # Select samples with highest vote entropy
        n_select = 10
        most_disagreed_idx = np.argsort(vote_entropies)[-n_select:]
        
        # Get selected samples
        selected_samples = unlabeled_data.iloc[most_disagreed_idx].copy()
        selected_entropies = vote_entropies[most_disagreed_idx]
        
        print(f"\nSelected {len(selected_samples)} most disagreed samples:")
        for i, (idx, row) in enumerate(selected_samples.iterrows()):
            entropy_val = selected_entropies[i]
            print(f"{i+1}. Vote Entropy: {entropy_val:.4f} - {row['text'][:80]}...")
        
        # Add to labeled data
        labeled_data = pd.concat([labeled_data, selected_samples], ignore_index=True)
        
        # Remove from unlabeled data
        unlabeled_data = unlabeled_data.drop(selected_samples.index).reset_index(drop=True)
        
        print(f"Updated pools: Labeled={len(labeled_data)}, Unlabeled={len(unlabeled_data)}")
    
    print("\nCommittee sampling demo completed!")
    return labeled_data, committee

def main():
    """Run both demos"""
    print("Arabic Text Classification - Active Learning Demo")
    print("(Using TF-IDF instead of SBERT for compatibility)")
    
    # Run uncertainty sampling demo
    labeled_data_1, classifier_1 = uncertainty_sampling_demo()
    
    # Run committee demo
    labeled_data_2, committee_2 = committee_demo()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Both active learning approaches have been demonstrated.")
    print("\nTo use SBERT embeddings and GPU acceleration:")
    print("1. Install dependencies: pip install sentence-transformers lightgbm")
    print("2. Fix any version conflicts with huggingface-hub")
    print("3. Run the full implementation in main.py")

if __name__ == "__main__":
    main()