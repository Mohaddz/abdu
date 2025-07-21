# Arabic Text Classification - Active Learning Pipelines

GPU-accelerated active learning pipelines for Arabic text classification using SBERT embeddings and multiple sampling strategies.

## Project Overview

This project implements two active learning approaches for binary Arabic text classification (security vs non-security):

1. **Uncertainty Sampling Pipeline**: Uses SBERT embeddings with a single classifier
2. **Committee-Based Sampling Pipeline**: Uses SBERT embeddings with multiple classifiers

## Features

- 🚀 **GPU-accelerated** processing with CUDA support
- 🤖 **SBERT embeddings** for Arabic text using multilingual models
- 📊 **Multiple sampling strategies**: uncertainty, margin, entropy, vote entropy, disagreement
- 🧠 **Committee learning** with multiple classifiers (LogisticRegression, SVM, LightGBM, RandomForest)
- 📝 **Modular design** with clean separation of concerns
- 📚 **Jupyter notebook** demo with detailed explanations

## Directory Structure

```
ActiveLearning-Abdu/
├── pipelines/
│   ├── __init__.py
│   ├── uncertainty_sampling.py    # Uncertainty sampling implementation
│   └── committee_sampling.py      # Committee-based sampling implementation
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # CSV data loading utilities
│   └── embeddings.py              # SBERT embedding generation
├── notebooks/
│   └── active_learning_demo.ipynb # Interactive Jupyter demo
├── data/                          # Data directory (created automatically)
├── main.py                        # Main execution script
└── requirements.txt               # Python dependencies
```

## Installation & Setup

### Prerequisites

- Python 3.11
- CUDA 12.6+ (for GPU acceleration)
- Conda environment recommended

### Environment Setup

```bash
# Create conda environment
conda create -n al_env python=3.11
conda activate al_env

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
pip install scikit-learn pandas numpy scipy jupyter

# Install SBERT (may require compatible huggingface-hub version)
pip install sentence-transformers>=2.7.0 huggingface-hub>=0.19.0

# Install LightGBM
pip install lightgbm
```

### Quick Start (Compatible Demo)

If you encounter dependency issues, run the simplified demo first:

```bash
python simple_demo.py
```

This uses TF-IDF instead of SBERT and demonstrates both active learning approaches without external model dependencies.

### Cloud Deployment

For cloud machines (Google Colab, AWS, Azure, etc.):

```bash
# Clone the repository
git clone <your-repo-url>
cd ActiveLearning-Abdu

# Install dependencies in cloud environment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Test GPU availability
python test_dependencies.py

# Run the demo
python main.py  # Full SBERT version
# OR
python simple_demo.py  # Compatible version
```

**Google Colab Setup:**
```python
!git clone <your-repo-url>
%cd ActiveLearning-Abdu
!pip install -r requirements.txt
!python main.py
```

### GPU-enabled LightGBM (Optional)

For GPU acceleration with LightGBM:

```bash
pip uninstall lightgbm
pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON
```

## Usage

### Method 1: Run the Main Script

```bash
python main.py
```

This will:
- Create sample Arabic data automatically
- Run both pipelines with demonstration data
- Show results and selected samples

### Method 2: Use Jupyter Notebook

```bash
jupyter notebook notebooks/active_learning_demo.ipynb
```

The notebook provides:
- Step-by-step pipeline execution
- Detailed explanations and visualizations
- Interactive exploration of results

### Method 3: Use with Your Own Data

```python
from utils.data_loader import DataLoader
from utils.embeddings import SBERTEmbedder
from pipelines.uncertainty_sampling import UncertaintySamplingPipeline

# Load your CSV data (must have 'text' and 'label' columns)
data_loader = DataLoader("path/to/your/arabic_data.csv")
embedder = SBERTEmbedder()

# Get initial split
labeled_texts, labeled_labels, unlabeled_texts, unlabeled_labels = data_loader.get_initial_split(
    initial_labeled_size=100
)

# Initialize pipeline
pipeline = UncertaintySamplingPipeline(embedder, classifier_type='logistic')
pipeline.initialize_pools(labeled_texts, labeled_labels, unlabeled_texts)

# Run active learning iteration
results = pipeline.run_active_learning_iteration(n_samples=20, strategy='least_confident')
```

## Data Format

Your CSV file should contain:
- `text`: Arabic text content
- `label`: Binary labels (0 for non-security, 1 for security)

Example:
```csv
text,label
"أمان المعلومات والحماية الرقمية",1
"الطقس اليوم مشمس وجميل",0
```

## Pipeline Details

### Uncertainty Sampling Pipeline

- **Embeddings**: SBERT (paraphrase-multilingual-MiniLM-L12-v2)
- **Classifiers**: LogisticRegression, SVM
- **Strategies**: 
  - `least_confident`: 1 - max(P(y|x))
  - `margin`: difference between top 2 predictions
  - `entropy`: prediction entropy

### Committee-Based Sampling Pipeline

- **Embeddings**: SBERT (paraphrase-multilingual-MiniLM-L12-v2)
- **Committee**: LogisticRegression, SVM, LightGBM, RandomForest
- **Strategies**:
  - `vote_entropy`: entropy of averaged predictions
  - `disagreement`: proportion of disagreeing classifiers

## Configuration Options

### Uncertainty Sampling
```python
pipeline = UncertaintySamplingPipeline(
    embedder=embedder,
    classifier_type='logistic',  # 'logistic' or 'svm'
    random_state=42
)
```

### Committee Sampling
```python
pipeline = CommitteeSamplingPipeline(
    embedder=embedder,
    use_gpu_lightgbm=True,  # Enable GPU LightGBM
    random_state=42
)
```

## GPU Acceleration

The project supports GPU acceleration for:
- ✅ **SBERT embeddings**: Automatic GPU utilization
- ✅ **LightGBM**: GPU training when available
- ✅ **PyTorch**: CUDA operations

Check GPU status:
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
```

## Performance Tips

1. **Batch Processing**: Adjust `batch_size` in embedding generation
2. **Initial Pool Size**: Start with 50-200 labeled samples
3. **Iteration Size**: Select 10-50 samples per iteration
4. **GPU Memory**: Monitor memory usage for large datasets

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in embedding generation
   - Use CPU fallback: `device='cpu'`

2. **LightGBM GPU Issues**:
   - Set `use_gpu_lightgbm=False`
   - Install CPU version: `pip install lightgbm`

3. **Arabic Text Encoding**:
   - Ensure CSV is saved with UTF-8 encoding
   - Use `encoding='utf-8'` when reading files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is available for educational and research purposes.