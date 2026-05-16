# Investment Research Desk Sentiment LoRA Held-out Evaluation

- Adapter: `models/investment-research-desk-lora-sentiment/20260515T123418Z/adapter`
- Accuracy: 0.8926
- Macro-F1: 0.8760
- Baseline accuracy delta: +0.1026
- Baseline Macro-F1 delta: +0.0989
- Output contract: `pass`

## Dataset Results

### financial_phrasebank

- Split: `test`
- Samples: 484
- Accuracy: 0.8740
- Macro-F1: 0.8631

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.9123 | 0.8525 | 0.8814 |
| neutral | 0.8626 | 0.9408 | 0.9000 |
| positive | 0.8860 | 0.7426 | 0.8080 |

### twitter_financial_news_sentiment

- Split: `validation`
- Samples: 2388
- Accuracy: 0.9112
- Macro-F1: 0.8890

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| bearish | 0.8629 | 0.8703 | 0.8666 |
| bullish | 0.8838 | 0.8484 | 0.8657 |
| neutral | 0.9298 | 0.9393 | 0.9346 |
