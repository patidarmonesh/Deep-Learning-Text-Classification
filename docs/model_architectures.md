# Model Architectures

Detailed documentation of all implemented model architectures.

---

## Table of Contents
1. [LSTM Models](#lstm-models)
2. [GRU Models](#gru-models)
3. [CNN Models](#cnn-models)
4. [BERT Model](#bert-model)

---

## LSTM Models

### 1. Simple LSTM

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Embedding Layer (300 dimensions)
    ↓
LSTM Layer (hidden_dim=256, num_layers=2)
    ↓
Dropout (p=0.5)
    ↓
Fully Connected Layer (256 → 4)
    ↓
Output (4 classes)
```

**Parameters:**
- Embedding dimension: 300
- Hidden dimension: 256
- Number of layers: 2
- Dropout rate: 0.5
- Total parameters: ~2.1M

**Performance:**
- Validation Accuracy: 88.88%
- Validation Loss: 0.4507
- Training Time: ~45 minutes

---

### 2. BiLSTM (Bidirectional LSTM)

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Embedding Layer (300 dimensions)
    ↓
Bidirectional LSTM (hidden_dim=256, num_layers=2)
    ↓ (forward + backward)
Concatenated Hidden States (512 dimensions)
    ↓
Dropout (p=0.5)
    ↓
Fully Connected Layer (512 → 4)
    ↓
Output (4 classes)
```

**Parameters:**
- Embedding dimension: 300
- Hidden dimension: 256 (per direction)
- Number of layers: 2
- Dropout rate: 0.5
- Total parameters: ~4.2M

**Performance:**
- Validation Accuracy: 88.76%
- Validation Loss: 0.5300
- Training Time: ~52 minutes

---

### 3. LSTM + Attention

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Embedding Layer (300 dimensions)
    ↓
LSTM Layer (hidden_dim=256, num_layers=2)
    ↓
Attention Mechanism
    ├── Attention Scores = tanh(W * hidden_states)
    ├── Attention Weights = softmax(attention_scores)
    └── Context Vector = Σ(attention_weights * hidden_states)
    ↓
Dropout (p=0.5)
    ↓
Fully Connected Layer (256 → 4)
    ↓
Output (4 classes)
```

**Attention Formula:**
```
e_i = tanh(W_a * h_i + b_a)
α_i = exp(e_i) / Σ exp(e_j)
c = Σ α_i * h_i
```

**Parameters:**
- Embedding dimension: 300
- Hidden dimension: 256
- Number of layers: 2
- Attention dimension: 256
- Dropout rate: 0.5
- Total parameters: ~2.3M

**Performance:**
- Validation Accuracy: 88.49%
- Validation Loss: 0.5848
- Training Time: ~48 minutes

---

### 4. BiLSTM + Attention

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Embedding Layer (300 dimensions)
    ↓
Bidirectional LSTM (hidden_dim=256, num_layers=2)
    ↓ (forward + backward concatenated)
Attention Mechanism (on 512-dim hidden states)
    ├── Attention Scores
    ├── Attention Weights
    └── Context Vector
    ↓
Dropout (p=0.5)
    ↓
Fully Connected Layer (512 → 4)
    ↓
Output (4 classes)
```

**Parameters:**
- Embedding dimension: 300
- Hidden dimension: 256 (per direction)
- Number of layers: 2
- Attention dimension: 512
- Dropout rate: 0.5
- Total parameters: ~4.4M

**Performance:**
- Validation Accuracy: 88.68%
- Validation Loss: 0.6037
- Training Time: ~55 minutes

---

## GRU Models

### 1. Simple GRU

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Embedding Layer (300 dimensions)
    ↓
GRU Layer (hidden_dim=256, num_layers=2)
    ↓
Dropout (p=0.5)
    ↓
Fully Connected Layer (256 → 4)
    ↓
Output (4 classes)
```

**GRU Cell Operations:**
```
Update Gate: z_t = σ(W_z * [h_{t-1}, x_t])
Reset Gate: r_t = σ(W_r * [h_{t-1}, x_t])
Candidate: h̃_t = tanh(W_h * [r_t ⊙ h_{t-1}, x_t])
Hidden State: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**Parameters:**
- Embedding dimension: 300
- Hidden dimension: 256
- Number of layers: 2
- Dropout rate: 0.5
- Total parameters: ~1.6M

**Performance:**
- Validation Accuracy: 88.35%
- Validation Loss: 0.5714
- Training Time: ~32 minutes

---

### 2. BiGRU

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Embedding Layer (300 dimensions)
    ↓
Bidirectional GRU (hidden_dim=256, num_layers=2)
    ↓ (forward + backward)
Concatenated Hidden States (512 dimensions)
    ↓
Dropout (p=0.5)
    ↓
Fully Connected Layer (512 → 4)
    ↓
Output (4 classes)
```

**Parameters:**
- Embedding dimension: 300
- Hidden dimension: 256 (per direction)
- Number of layers: 2
- Dropout rate: 0.5
- Total parameters: ~3.2M

**Performance:**
- Validation Accuracy: 88.44%
- Validation Loss: 0.5919
- Training Time: ~38 minutes

---

### 3. GRU + Attention

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Embedding Layer (300 dimensions)
    ↓
GRU Layer (hidden_dim=256, num_layers=2)
    ↓
Attention Mechanism
    ├── Attention Scores
    ├── Attention Weights
    └── Context Vector
    ↓
Dropout (p=0.5)
    ↓
Fully Connected Layer (256 → 4)
    ↓
Output (4 classes)
```

**Parameters:**
- Embedding dimension: 300
- Hidden dimension: 256
- Number of layers: 2
- Attention dimension: 256
- Dropout rate: 0.5
- Total parameters: ~1.8M

**Performance:**
- Validation Accuracy: 88.24%
- Validation Loss: 0.6456
- Training Time: ~35 minutes

---

### 4. BiGRU + Attention

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Embedding Layer (300 dimensions)
    ↓
Bidirectional GRU (hidden_dim=256, num_layers=2)
    ↓
Attention Mechanism (on 512-dim hidden states)
    ├── Attention Scores
    ├── Attention Weights
    └── Context Vector
    ↓
Dropout (p=0.5)
    ↓
Fully Connected Layer (512 → 4)
    ↓
Output (4 classes)
```

**Parameters:**
- Embedding dimension: 300
- Hidden dimension: 256 (per direction)
- Number of layers: 2
- Attention dimension: 512
- Dropout rate: 0.5
- Total parameters: ~3.4M

**Performance:**
- Validation Accuracy: 88.44%
- Validation Loss: 0.6539
- Training Time: ~41 minutes

---

## CNN Models

### 1. Hierarchical CNN

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Embedding Layer (300 dimensions)
    ↓
Word-level Convolutions
    ├── Conv1D (filters=100, kernel_size=3)
    ├── Conv1D (filters=100, kernel_size=4)
    └── Conv1D (filters=100, kernel_size=5)
    ↓
Max Pooling (per filter)
    ↓
Concatenate (300 dimensions)
    ↓
Sentence-level Convolution
    └── Conv1D (filters=128, kernel_size=3)
    ↓
Max Pooling
    ↓
Fully Connected Layers
    ├── FC (128 → 64)
    ├── ReLU + Dropout (0.5)
    └── FC (64 → 4)
    ↓
Output (4 classes)
```

**Parameters:**
- Embedding dimension: 300
- Filter sizes: [3, 4, 5]
- Number of filters: 100 each
- Dropout rate: 0.5
- Total parameters: ~1.8M

**Performance:**
- Test Accuracy: 89.21%
- Test Loss: 0.8725
- Training Time: ~28 minutes

---

### 2. Strided CNN

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Embedding Layer (300 dimensions)
    ↓
Parallel Convolutional Layers
    ├── Conv1D (filters=128, kernel_size=3, stride=1)
    ├── Conv1D (filters=128, kernel_size=4, stride=1)
    └── Conv1D (filters=128, kernel_size=5, stride=1)
    ↓
Max Pooling (per convolution)
    ↓
Concatenate (384 dimensions)
    ↓
Dropout (p=0.5)
    ↓
Fully Connected Layer (384 → 4)
    ↓
Output (4 classes)
```

**Parameters:**
- Embedding dimension: 300
- Filter sizes: [3, 4, 5]
- Number of filters: 128 each
- Stride: 1
- Dropout rate: 0.5
- Total parameters: ~1.5M

**Performance:**
- Test Accuracy: 88.57%
- Test Loss: 0.8104
- Training Time: ~25 minutes

---

### 3. Multi-Channel CNN

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Dual Embedding Channels
    ├── Static Channel (frozen Word2Vec)
    └── Non-static Channel (trainable)
    ↓
Parallel Convolutions (per channel)
    ├── Conv1D (filters=100, kernel_size=3)
    ├── Conv1D (filters=100, kernel_size=4)
    └── Conv1D (filters=100, kernel_size=5)
    ↓
Max Pooling (per filter per channel)
    ↓
Concatenate (600 dimensions)
    ↓
Dropout (p=0.5)
    ↓
Fully Connected Layer (600 → 4)
    ↓
Output (4 classes)
```

**Parameters:**
- Embedding dimension: 300 (per channel)
- Filter sizes: [3, 4, 5]
- Number of filters: 100 each (per channel)
- Number of channels: 2
- Dropout rate: 0.5
- Total parameters: ~2.0M

**Performance:**
- Test Accuracy: 89.93%
- Test Loss: 0.4585
- Training Time: ~30 minutes

---

### 4. Dynamic Pooling CNN

**Architecture:**
```
Input Layer (Word Embeddings)
    ↓
Embedding Layer (300 dimensions)
    ↓
Convolutional Layers
    ├── Conv1D (filters=256, kernel_size=3)
    ├── ReLU
    └── BatchNorm1d
    ↓
Dynamic K-Max Pooling (k=top 3)
    ↓
Flatten
    ↓
Fully Connected Layers
    ├── FC (768 → 256)
    ├── ReLU + Dropout (0.5)
    └── FC (256 → 4)
    ↓
Output (4 classes)
```

**Parameters:**
- Embedding dimension: 300
- Number of filters: 256
- Kernel size: 3
- K-max pooling: top 3
- Dropout rate: 0.5
- Total parameters: ~2.1M

**Performance:**
- Test Accuracy: 90.38%
- Test Loss: 0.3204
- Training Time: ~33 minutes

---

## BERT Model

### Architecture

**Base Model:** `bert-base-uncased`

```
Input Layer (Token IDs)
    ↓
BERT Tokenizer (WordPiece)
    ├── Input IDs
    ├── Attention Mask
    └── Token Type IDs
    ↓
BERT Encoder (12 layers)
    ├── Multi-Head Self-Attention (12 heads)
    ├── Layer Normalization
    ├── Feed-Forward Network
    └── Residual Connections
    ↓
[CLS] Token Representation (768 dimensions)
    ↓
Dropout (p=0.1)
    ↓
Classification Head (768 → 4)
    ↓
Output (4 classes)
```

**BERT Encoder Specifications:**
- Number of layers: 12
- Hidden size: 768
- Attention heads: 12
- Intermediate size: 3072
- Vocabulary size: 30,522

**Fine-tuning Configuration:**
- Max sequence length: 128
- Batch size: 16
- Learning rate: 2e-5
- Warmup steps: 500
- Weight decay: 0.01
- Epochs: 3

**Parameters:**
- Total parameters: ~110M
- Trainable parameters: ~110M (all layers fine-tuned)

**Performance:**
- Test Accuracy: 92.79%
- Test Precision: 92.86%
- Test Recall: 92.79%
- Test F1-Score: 92.79%
- Training Time: 97.31 minutes

---

## Comparison Summary

| Model Type | Complexity | Parameters | Speed | Accuracy |
|------------|-----------|------------|-------|----------|
| **LSTM** | Medium | 2-4M | Slow | 88-89% |
| **GRU** | Medium | 1.6-3.4M | Medium | 88-88.5% |
| **CNN** | Low-Medium | 1.5-2.1M | Fast | 88-90% |
| **BERT** | Very High | 110M | Very Slow | 92-93% |

**Key Takeaway:** CNNs offer best efficiency-performance trade-off, while BERT provides best absolute performance at the cost of computational resources.
