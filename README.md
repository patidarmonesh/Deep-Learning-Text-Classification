# Advanced Deep Learning for Text Classification

<div align="center">

> A comprehensive implementation and benchmarking of state-of-the-art deep learning architectures for multiclass text classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Table of Contents
- [Overview](#overview)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Training Visualizations](#training-visualizations)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Key Learnings](#key-learnings)
- [Applications](#applications)
- [Future Work](#future-work)
- [Contact](#contact)
- [Citation](#citation)

---

## Overview

This project explores and compares **11 different deep learning architectures** for text classification on a multiclass news dataset. The implementation demonstrates the evolution of NLP models from recurrent architectures to modern transformers.

### Research Questions

1. How do sequential models (LSTM/GRU) compare with convolutional approaches for text classification?
2. What is the impact of attention mechanisms and bidirectionality on model performance?
3. How effective is transfer learning with pre-trained transformers compared to training from scratch?
4. What are the trade-offs between model complexity, training time, and accuracy?

### Dataset

| Property | Value |
|----------|-------|
| Task | 4-class text classification |
| Training samples | ~76,000 |
| Test samples | ~7,600 |
| Classes | World, Sports, Business, Sci/Tech |
| Average text length | ~42 words |

---

## Models Implemented

### 1. Recurrent Neural Networks

<table>
<tr>
<td width="50%">

#### LSTM Variants (4 models)
- **Simple LSTM**: Unidirectional LSTM for sequential text processing
- **BiLSTM**: Bidirectional LSTM capturing forward and backward context
- **LSTM + Attention**: LSTM with attention mechanism to focus on important tokens
- **BiLSTM + Attention**: Combines bidirectional processing with attention

**Key Insight**: Simple LSTM achieved best performance (88.88%) among RNN models.

</td>
<td width="50%">

#### GRU Variants (4 models)
- **Simple GRU**: Efficient alternative to LSTM with fewer parameters
- **BiGRU**: Bidirectional GRU for contextual understanding
- **GRU + Attention**: GRU enhanced with attention mechanism
- **BiGRU + Attention**: Full bidirectional GRU with attention

**Key Insight**: GRU models trained ~30% faster than LSTM with comparable accuracy.

</td>
</tr>
</table>

---

### 2. Convolutional Neural Networks (4 models)

<table>
<tr>
<td width="50%">

#### Hierarchical CNN
- Word-level and sentence-level convolutions
- Captures compositional semantics
- **Test accuracy**: 89.21%

#### Strided CNN
- Parallel filters with different kernel sizes
- Efficient n-gram feature extraction
- **Test accuracy**: 88.57%

</td>
<td width="50%">

#### Multi-Channel CNN
- Static and non-static embedding channels
- Balances transfer learning with adaptation
- **Test accuracy**: 89.93%

#### Dynamic Pooling CNN
- Adaptive pooling based on input length
- Best performing CNN variant
- **Test accuracy**: 90.38%

</td>
</tr>
</table>

**Key Insight**: CNNs provide significantly faster training and inference compared to RNNs while maintaining competitive accuracy.

---

### 3. Transformer Model

#### BERT Fine-tuning
- **Base model**: `bert-base-uncased` (110M parameters)
- **Training**: 3 epochs with gradient accumulation
- **Architecture**: Bidirectional transformer with 12 layers
- **Pre-training**: 3.3B words

**Key Insight**: BERT achieved state-of-the-art results (92.79% accuracy), demonstrating the power of large-scale pre-training.

---

## Results

### Model Performance Comparison

| Model | Accuracy | Loss | Training Time | Parameters |
|-------|----------|------|---------------|------------|
| LSTM | 88.88% | 0.4507 | ~45 min | 2.1M |
| BiLSTM | 88.76% | 0.5300 | ~52 min | 4.2M |
| LSTM + Attention | 88.49% | 0.5848 | ~48 min | 2.3M |
| BiLSTM + Attention | 88.68% | 0.6037 | ~55 min | 4.4M |
| GRU | 88.35% | 0.5714 | ~32 min | 1.6M |
| BiGRU | 88.44% | 0.5919 | ~38 min | 3.2M |
| GRU + Attention | 88.24% | 0.6456 | ~35 min | 1.8M |
| BiGRU + Attention | 88.44% | 0.6539 | ~41 min | 3.4M |
| Hierarchical CNN | 89.21% | 0.8725 | ~28 min | 1.8M |
| Strided CNN | 88.57% | 0.8104 | ~25 min | 1.5M |
| Multi-Channel CNN | 89.93% | 0.4585 | ~30 min | 2.0M |
| Dynamic Pooling CNN | 90.38% | 0.3204 | ~33 min | 2.1M |
| **BERT (fine-tuned)** | **92.79%** | **0.2440** | **97 min** | **110M** |

### Key Findings

<table>
<tr>
<td width="20%"><b>Best Overall</b></td>
<td width="80%">BERT achieves highest accuracy (92.79%) due to massive pre-training</td>
</tr>
<tr>
<td width="20%"><b>Best RNN</b></td>
<td width="80%">Simple LSTM performs best among recurrent models (88.88%)</td>
</tr>
<tr>
<td width="20%"><b>Best CNN</b></td>
<td width="80%">Dynamic Pooling CNN achieves 90.38% with fast training</td>
</tr>
<tr>
<td width="20%"><b>Fastest</b></td>
<td width="80%">Strided CNN trains in just 25 minutes</td>
</tr>
<tr>
<td width="20%"><b>Best Efficiency</b></td>
<td width="80%">Multi-Channel CNN offers 89.93% accuracy with only 2M parameters</td>
</tr>
</table>

---

## Training Visualizations

### LSTM Models Training Dynamics

<div align="center">
<img src="results/lstm comparisions.png" alt="LSTM Training Comparison" width="100%">
</div>

The visualization shows training and validation performance across all LSTM variants:
- **Training Loss**: All models converge smoothly, with attention variants showing slightly higher final loss
- **Validation Loss**: Simple LSTM maintains lowest validation loss, suggesting better generalization
- **Training Accuracy**: All models achieve >95% training accuracy by epoch 10
- **Validation Accuracy**: Simple LSTM peaks at 88.88%, outperforming more complex variants

---

### GRU Models Training Dynamics

<div align="center">
<img src="results/gru comparision.png" alt="GRU Training Comparison" width="100%">
</div>

The visualization demonstrates GRU models' training characteristics:
- **Training Loss**: GRU models converge faster than LSTM counterparts
- **Validation Loss**: All variants show similar convergence patterns with slight overfitting in later epochs
- **Training Accuracy**: Rapid initial learning, reaching >97% by epoch 10
- **Validation Accuracy**: BiGRU and BiGRU+Attention achieve 88.44%, with slight fluctuations

**Observation**: The validation accuracy curves show that simpler models (LSTM, GRU) generalize better than their bidirectional and attention-enhanced counterparts, suggesting that added complexity doesn't always improve performance on this dataset.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for BERT)
- 8GB+ RAM

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/patidarmonesh/Deep-Learning-Text-Classification.git
cd Deep-Learning-Text-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"
```

---

## Usage

### Running the Notebook

```bash
# Start Jupyter notebook
jupyter notebook Deep_Learning_Text_Classification.ipynb

# Or use Google Colab for free GPU
# Upload notebook to Google Drive and open with Colab
```

### Quick Start

The notebook is organized sequentially for easy navigation:

<table>
<tr>
<th>Section</th>
<th>Cells</th>
<th>Description</th>
</tr>
<tr>
<td>Data Loading</td>
<td>1-15</td>
<td>Dataset loading and preprocessing pipeline</td>
</tr>
<tr>
<td>LSTM Models</td>
<td>16-33</td>
<td>Implementation and training of 4 LSTM variants</td>
</tr>
<tr>
<td>GRU Models</td>
<td>34-52</td>
<td>Implementation and training of 4 GRU variants</td>
</tr>
<tr>
<td>CNN Models</td>
<td>53-103</td>
<td>Implementation of 4 CNN architectures</td>
</tr>
<tr>
<td>BERT Fine-tuning</td>
<td>104-131</td>
<td>Transfer learning with pre-trained BERT</td>
</tr>
</table>

### Custom Training Example

```python
# Example: Train a custom BiLSTM model
model = BiLSTMClassifier(
    embedding_dim=300,
    hidden_dim=256,
    num_layers=2,
    num_classes=4,
    dropout=0.5
)

# Train the model
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    learning_rate=0.001
)
```

---

## Project Structure

```
Deep-Learning-Text-Classification/
│
├── README.md                                    # Project documentation
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore rules
│
├── Deep_Learning_Text_Classification.ipynb     # Main notebook
│
├── docs/                                        # Documentation
│   ├── model_architectures.md                  # Detailed architecture descriptions
│   ├── hyperparameters.md                      # Hyperparameter configurations
│   └── analysis.md                             # In-depth analysis
│
├── results/                                     # Results and visualizations
│   ├── lstm-comparisions.jpg                   # LSTM training curves
│   ├── gru-comparision.jpg                     # GRU training curves
│   └── extracted_results.txt                   # Performance summary
│
└── data/                                        # Dataset directory
    └── README.md                               # Data description
```

---

## Technical Details

### Embeddings
- **Pre-trained**: Google News Word2Vec (300 dimensions)
- **Vocabulary**: 3M words
- **Strategy**: Frozen embeddings to preserve semantic relationships

### Training Configuration

<table>
<tr>
<td width="50%">

**Optimization**
- Optimizer: Adam
- Beta parameters: (0.9, 0.999)
- Learning rate: 0.001 (RNN/CNN), 2e-5 (BERT)
- Scheduler: ReduceLROnPlateau
- Gradient clipping: Max norm = 1.0

</td>
<td width="50%">

**Training Setup**
- Batch size: 32 (RNN/CNN), 16 (BERT)
- Max sequence length: 256 tokens
- Early stopping: Patience of 5 epochs
- Loss function: CrossEntropyLoss
- Epochs: 10 (RNN/CNN), 3 (BERT)

</td>
</tr>
</table>

### Data Preprocessing Pipeline
1. Text lowercasing
2. Special character removal
3. Tokenization
4. Stopword removal (optional)
5. Padding/truncation to fixed length

### Hardware Specifications
- **GPU**: NVIDIA Tesla T4 (Google Colab)
- **RAM**: 12GB
- **Storage**: 100GB
- **CUDA**: 11.8
- **PyTorch**: 2.0.1

---

## Key Learnings

### 1. Architecture Trade-offs

<table>
<tr>
<th>Model Type</th>
<th>Strengths</th>
<th>Weaknesses</th>
</tr>
<tr>
<td><b>RNNs</b></td>
<td>Excel at sequence modeling, capture long-range dependencies</td>
<td>Computationally expensive, slow training</td>
</tr>
<tr>
<td><b>CNNs</b></td>
<td>Fast training, parallelizable, efficient for local patterns</td>
<td>May miss long-range dependencies</td>
</tr>
<tr>
<td><b>Transformers</b></td>
<td>State-of-the-art performance, bidirectional context</td>
<td>Require significant compute resources, large memory</td>
</tr>
</table>

### 2. Attention Mechanisms
- Attention improved some LSTM models but not consistently
- Simple LSTM without attention performed best (88.88%)
- Attention adds computational overhead that may not always justify the gain
- Attention is more effective when combined with proper regularization

### 3. Bidirectionality
- BiLSTM and BiGRU models showed mixed results
- Simple unidirectional models sometimes outperformed bidirectional variants
- Bidirectionality doubles parameters and training time
- May lead to overfitting on smaller datasets

### 4. Transfer Learning
- BERT's pre-training provided 2.4% boost over best from-scratch model
- Only 3 epochs of fine-tuning sufficient for convergence
- Demonstrates value of large-scale pre-training on massive corpora
- Transfer learning is crucial for achieving state-of-the-art results

### 5. GRU vs LSTM
- GRU models train 25-30% faster than LSTM
- LSTM achieved slightly better accuracy (88.88% vs 88.44%)
- GRU offers better efficiency-performance trade-off
- GRU is preferable when training time is a constraint

---

## Applications

This research can be applied to various real-world scenarios:

<table>
<tr>
<td width="50%">

**Content Management**
- News categorization
- Automated content tagging
- Document organization
- Content recommendation

**Business Intelligence**
- Customer feedback classification
- Market research analysis
- Competitive intelligence
- Trend detection

</td>
<td width="50%">

**Customer Service**
- Ticket routing and prioritization
- Sentiment analysis
- Query classification
- Automated response suggestion

**Compliance & Legal**
- Legal document classification
- Regulatory compliance checking
- Contract categorization
- Risk assessment

</td>
</tr>
</table>

---

## Future Work

- [ ] Implement DistilBERT and RoBERTa for comparison
- [ ] Add ensemble methods combining multiple models
- [ ] Experiment with different attention mechanisms (multi-head, self-attention)
- [ ] Deploy best model as REST API
- [ ] Add model explainability with LIME or SHAP
- [ ] Try data augmentation techniques (back-translation, synonym replacement)
- [ ] Explore few-shot learning approaches
- [ ] Implement cross-validation for more robust evaluation
- [ ] Add confusion matrices and per-class analysis
- [ ] Benchmark inference speed on different hardware

---

## Contributing

Contributions are welcome! Please feel free to:

- Report bugs or issues
- Suggest new model architectures
- Improve documentation
- Add new features
- Submit pull requests

For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Monesh Patidar**  
IIT Kanpur

<div align="left">

[![GitHub](https://img.shields.io/badge/GitHub-patidarmonesh-181717?style=flat&logo=github)](https://github.com/patidarmonesh)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Monesh_Patidar-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/monesh-patidar-056763283/)
[![Email](https://img.shields.io/badge/Email-moeshp23@iitk.ac.in-D14836?style=flat&logo=gmail)](mailto:moeshp23@iitk.ac.in)

</div>

---

## Acknowledgments

This project was made possible by:

- **Google's Word2Vec** pre-trained embeddings for semantic word representations
- **Hugging Face Transformers** library for BERT implementation
- **PyTorch** framework for deep learning development
- **Google Colab** for providing free GPU resources
- **IIT Kanpur** for research support and guidance

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{patidar2026_text_classification,
  author = {Monesh Patidar},
  title = {Advanced Deep Learning for Text Classification: A Comparative Study of RNNs, CNNs, and Transformers},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/patidarmonesh/Deep-Learning-Text-Classification}}
}
```

---

<div align="center">

**Note**: This project was developed as part of deep learning research at IIT Kanpur.

---

**If you find this project useful, please consider giving it a star ⭐**

---

Made with dedication by [Monesh Patidar](https://github.com/patidarmonesh)

</div>
