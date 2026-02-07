# Hyperparameters

Complete documentation of hyperparameters used for all models.

---

## Table of Contents
1. [Common Hyperparameters](#common-hyperparameters)
2. [LSTM Models](#lstm-models)
3. [GRU Models](#gru-models)
4. [CNN Models](#cnn-models)
5. [BERT Model](#bert-model)

---

## Common Hyperparameters

### Data Preprocessing
- **Max sequence length**: 256 tokens
- **Vocabulary size**: 20,000 most frequent words
- **Padding**: Post-padding with zeros
- **Truncation**: Post-truncation for longer sequences

### Training Configuration
- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Beta parameters**: β1 = 0.9, β2 = 0.999
- **Epsilon**: 1e-8
- **Learning rate scheduler**: ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 3 epochs
  - Min learning rate: 1e-6
- **Early stopping patience**: 5 epochs
- **Gradient clipping**: Max norm = 1.0

### Hardware Configuration
- **Device**: NVIDIA Tesla T4 GPU (Google Colab)
- **Memory**: 12GB GPU RAM
- **CUDA version**: 11.8
- **PyTorch version**: 2.0.1

---

## LSTM Models

### Simple LSTM
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'hidden_dim': 256,
    'num_layers': 2,
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,

    # Data
    'max_length': 256,
    'train_size': 76000,
    'val_size': 7600,
}
```

### BiLSTM
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'hidden_dim': 256,
    'num_layers': 2,
    'bidirectional': True,
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,
}
```

### LSTM + Attention
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'hidden_dim': 256,
    'num_layers': 2,
    'attention_dim': 256,
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,
}
```

### BiLSTM + Attention
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'hidden_dim': 256,
    'num_layers': 2,
    'bidirectional': True,
    'attention_dim': 512,
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,
}
```

---

## GRU Models

### Simple GRU
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'hidden_dim': 256,
    'num_layers': 2,
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,
}
```

### BiGRU
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'hidden_dim': 256,
    'num_layers': 2,
    'bidirectional': True,
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,
}
```

### GRU + Attention
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'hidden_dim': 256,
    'num_layers': 2,
    'attention_dim': 256,
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,
}
```

### BiGRU + Attention
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'hidden_dim': 256,
    'num_layers': 2,
    'bidirectional': True,
    'attention_dim': 512,
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,
}
```

---

## CNN Models

### Hierarchical CNN
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'num_filters': 100,
    'filter_sizes': [3, 4, 5],
    'sentence_filters': 128,
    'sentence_kernel': 3,
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,
}
```

### Strided CNN
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'num_filters': 128,
    'filter_sizes': [3, 4, 5],
    'stride': 1,
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,
}
```

### Multi-Channel CNN
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'num_channels': 2,
    'num_filters': 100,
    'filter_sizes': [3, 4, 5],
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,
}
```

### Dynamic Pooling CNN
```python
hyperparameters = {
    # Architecture
    'embedding_dim': 300,
    'num_filters': 256,
    'kernel_size': 3,
    'k_max_pooling': 3,
    'hidden_dim': 256,
    'num_classes': 4,
    'dropout': 0.5,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'weight_decay': 0.0,
}
```

---

## BERT Model

### Configuration
```python
hyperparameters = {
    # Model
    'model_name': 'bert-base-uncased',
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'intermediate_size': 3072,
    'max_position_embeddings': 512,
    'vocab_size': 30522,

    # Fine-tuning
    'max_length': 128,
    'num_classes': 4,
    'classifier_dropout': 0.1,

    # Training
    'batch_size': 16,
    'gradient_accumulation_steps': 2,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,

    # Scheduler
    'scheduler_type': 'linear',
    'lr_scheduler_warmup_ratio': 0.1,
}
```

### Training Strategy
- **Warmup**: Linear warmup for first 10% of training steps
- **Learning rate decay**: Linear decay after warmup
- **Gradient accumulation**: Effective batch size = 32 (16 × 2)
- **Mixed precision**: FP16 training for faster computation

---

## Hyperparameter Tuning Notes

### What Was Tuned
1. **Learning rate**: Tested [0.0001, 0.001, 0.01]
   - Best: 0.001 for RNN/CNN, 2e-5 for BERT
2. **Batch size**: Tested [16, 32, 64]
   - Best: 32 for RNN/CNN, 16 for BERT
3. **Dropout**: Tested [0.3, 0.5, 0.7]
   - Best: 0.5 for most models
4. **Hidden dimensions**: Tested [128, 256, 512]
   - Best: 256 for balanced performance
5. **Number of layers**: Tested [1, 2, 3]
   - Best: 2 layers for RNNs

### What Was Not Tuned
- Embedding dimension (fixed at 300 for Word2Vec)
- Optimizer (Adam worked well for all models)
- Loss function (CrossEntropy standard for classification)

### Observations
- Higher dropout (0.5-0.7) prevented overfitting in deep models
- Bidirectional models required lower learning rates
- CNN models were less sensitive to hyperparameters
- BERT required careful learning rate tuning

---

## Reproducibility

### Random Seeds
```python
import torch
import numpy as np
import random

seed = 42

# Python
random.seed(seed)

# Numpy
np.random.seed(seed)

# PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# CuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Data Splits
- Training set: 90% of data
- Validation set: 10% of data
- Test set: Separate held-out set

### Initialization
- Embedding layer: Pre-trained Word2Vec (frozen)
- Linear layers: Xavier uniform initialization
- LSTM/GRU: Default PyTorch initialization
- CNN: Kaiming normal initialization

---

## Computational Requirements

### Memory Usage (GPU)
- LSTM models: ~2-4 GB
- GRU models: ~1.5-3 GB
- CNN models: ~1-2 GB
- BERT: ~6-8 GB

### Training Time (per epoch)
- LSTM: ~4-5 minutes
- GRU: ~3-4 minutes
- CNN: ~2-3 minutes
- BERT: ~30-35 minutes

### Inference Time (per batch of 32)
- LSTM: ~50-70 ms
- GRU: ~40-60 ms
- CNN: ~20-30 ms
- BERT: ~100-150 ms

---

## Best Practices

1. **Start with lower learning rates** for bidirectional models
2. **Use gradient clipping** to prevent exploding gradients in RNNs
3. **Monitor validation loss** for early stopping
4. **Use learning rate scheduling** to improve convergence
5. **Apply dropout** to prevent overfitting in deep models
6. **Freeze embeddings** to preserve pre-trained semantic knowledge
7. **Use appropriate batch sizes** based on GPU memory
8. **Accumulate gradients** for BERT when GPU memory is limited
