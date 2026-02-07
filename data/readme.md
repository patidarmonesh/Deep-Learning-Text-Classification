# Dataset Information

## Overview
This dataset contains news articles classified into 4 categories.

## Statistics
- Training samples: 76,000
- Test samples: 7,600
- Total: 83,600 articles
- Classes: 4 (World, Sports, Business, Sci/Tech)
- Average article length: ~42 words
- Vocabulary size: ~20,000 unique words

## Data Format
Parquet files with columns:
- text: Article content
- label: Category (0-3)

## Preprocessing Applied
1. Text lowercasing
2. Special character removal
3. Tokenization
4. Padding/truncation to 256 tokens

## Source
[https://huggingface.co/datasets/Exploration-Lab/CS779-Fall25/tree/refs%2Fconvert%2Fparquet/Deep-learning-assignment]
