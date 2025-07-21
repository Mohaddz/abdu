#!/usr/bin/env python3

import pandas as pd
import numpy as np

def explain_active_learning():
    """
    Simple explanation of how active learning works with labeled/unlabeled data
    """
    print("=" * 80)
    print("ACTIVE LEARNING EXPLAINED - HOW IT HANDLES LABELED/UNLABELED DATA")
    print("=" * 80)
    
    # Example dataset - imagine you have 1000 Arabic texts but only 50 are labeled
    total_samples = 1000
    labeled_samples = 50
    unlabeled_samples = total_samples - labeled_samples
    
    print(f"\n🎯 THE PROBLEM:")
    print(f"   • You have {total_samples} Arabic texts")
    print(f"   • Only {labeled_samples} are labeled (expensive - human work)")
    print(f"   • {unlabeled_samples} are unlabeled (cheap - just raw text)")
    print(f"   • You want to label more, but which ones should you choose?")
    
    print(f"\n💡 ACTIVE LEARNING SOLUTION:")
    print(f"   Instead of randomly picking texts to label...")
    print(f"   → Train a model on your {labeled_samples} labeled samples")
    print(f"   → Use model to find MOST UNCERTAIN unlabeled samples")
    print(f"   → Label those uncertain samples (human work)")
    print(f"   → Add newly labeled samples to training set")
    print(f"   → Repeat process")
    
    print(f"\n📊 HOW THE SCRIPT WORKS:")
    
    # Simulate the process step by step
    print(f"\n   STEP 1 - Initial State:")
    print(f"   ├── Labeled Pool: {labeled_samples} samples ✅")
    print(f"   └── Unlabeled Pool: {unlabeled_samples} samples ❓")
    
    print(f"\n   STEP 2 - Train Model:")
    print(f"   ├── Use {labeled_samples} labeled samples to train classifier")
    print(f"   └── Model learns patterns: 'security words' = 1, 'other' = 0")
    
    print(f"\n   STEP 3 - Find Uncertain Samples:")
    print(f"   ├── Run model on {unlabeled_samples} unlabeled samples")
    print(f"   ├── Find samples where model is MOST CONFUSED")
    print(f"   └── Example: 'هذا النص معقد' → Model: 51% security, 49% not security ❓")
    
    print(f"\n   STEP 4 - Human Labels Selected Samples:")
    print(f"   ├── Show human the most uncertain 20 samples")
    print(f"   ├── Human labels them: 'هذا النص معقد' → 0 (not security)")
    print(f"   └── Add to labeled pool")
    
    print(f"\n   STEP 5 - Update Pools:")
    print(f"   ├── Labeled Pool: {labeled_samples + 20} samples ✅ (+20)")
    print(f"   └── Unlabeled Pool: {unlabeled_samples - 20} samples ❓ (-20)")
    
    print(f"\n   🔄 REPEAT until satisfied with model performance")

def show_script_behavior():
    """Show exactly how the script handles your data"""
    
    print(f"\n" + "=" * 80)
    print("HOW THE SCRIPT HANDLES YOUR CSV DATA")
    print("=" * 80)
    
    print(f"\n📁 YOUR CSV FILE:")
    sample_data = {
        'text': [
            'تسريب البيانات الشخصية',
            'الطقس اليوم جميل', 
            'اختراق أنظمة الحماية',
            'وصفة الطبخ العربية',
            'فيروسات الحاسوب'
        ],
        'label': [1, 0, 1, 0, 1]
    }
    
    df = pd.DataFrame(sample_data)
    print(df.to_string(index=False))
    
    print(f"\n🎭 SCRIPT SIMULATION PROCESS:")
    print(f"\n   1️⃣ Script takes your FULLY LABELED dataset")
    print(f"   2️⃣ PRETENDS some are unlabeled (for demonstration)")
    print(f"   3️⃣ Runs active learning to 'discover' which to label")
    print(f"   4️⃣ REVEALS the true labels (simulates human work)")
    
    print(f"\n📋 EXAMPLE EXECUTION:")
    
    # Show what actually happens
    initial_labeled = [0, 2]  # Indices 0 and 2 start as "labeled"
    initial_unlabeled = [1, 3, 4]  # Indices 1, 3, 4 are "unlabeled"
    
    print(f"\n   Initial Labeled Pool (known labels):")
    for i in initial_labeled:
        print(f"   ├── '{df.iloc[i]['text']}' → {df.iloc[i]['label']}")
    
    print(f"\n   Initial Unlabeled Pool (hidden labels):")
    for i in initial_unlabeled:
        print(f"   ├── '{df.iloc[i]['text']}' → ❓ (label hidden)")
    
    print(f"\n   Active Learning Process:")
    print(f"   ├── Train model on labeled samples")
    print(f"   ├── Model predicts unlabeled samples")
    print(f"   ├── Find most uncertain: 'الطقس اليوم جميل' (maybe 55% not security)")
    print(f"   ├── 'Label' this sample → Reveal true label: 0")
    print(f"   └── Add to labeled pool, continue...")

def real_world_usage():
    """Explain how to use with real unlabeled data"""
    
    print(f"\n" + "=" * 80)
    print("USING WITH YOUR REAL UNLABELED DATA")
    print("=" * 80)
    
    print(f"\n🎯 IF YOU HAVE REAL UNLABELED DATA:")
    print(f"\n   Your CSV should look like:")
    print(f"   ┌─────────────────────────┬───────┐")
    print(f"   │ text                    │ label │")
    print(f"   ├─────────────────────────┼───────┤")
    print(f"   │ تسريب البيانات          │   1   │  ← Labeled")
    print(f"   │ الطقس اليوم جميل        │   0   │  ← Labeled") 
    print(f"   │ هذا نص لم يُصنف بعد     │  -1   │  ← Unlabeled (use -1)")
    print(f"   │ نص آخر غير مصنف        │  -1   │  ← Unlabeled")
    print(f"   └─────────────────────────┴───────┘")
    
    print(f"\n   🔄 REAL ACTIVE LEARNING PROCESS:")
    print(f"   1️⃣ Script finds most uncertain unlabeled samples")
    print(f"   2️⃣ Shows them to YOU (human annotator)")
    print(f"   3️⃣ YOU manually label them (0 or 1)")
    print(f"   4️⃣ Script adds your labels to training data")
    print(f"   5️⃣ Repeat until model is good enough")
    
    print(f"\n   💡 REPLACE THIS LINE IN THE SCRIPT:")
    print(f"   # Instead of:")
    print(f"   simulated_labels = [0, 1, 0]  # Script guesses")
    print(f"   ")
    print(f"   # Use this:")
    print(f"   real_labels = input_human_labels(selected_texts)  # You label")

def main():
    explain_active_learning()
    show_script_behavior() 
    real_world_usage()
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Active Learning = Smart way to choose which data to label")
    print(f"✅ Script can handle both simulation (all labeled) and real unlabeled data")
    print(f"✅ Goal: Get better model with less human labeling work")
    print(f"✅ Your job: Label the samples the algorithm thinks are most important")

if __name__ == "__main__":
    main()