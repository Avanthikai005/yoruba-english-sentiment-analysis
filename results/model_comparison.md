# Model Comparison Summary

| Dataset | Model | Accuracy | Precision | Recall | F1 Score | Parameters | Best Epoch | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SST-2 (English)** | TextCNN | 92.12% | 92.84% | 93.04% | **92.94%** | 2,821,801 | 20 | 1430.3s |
| **SST-2 (English)** | CNN-BiLSTM-Attention | 90.16% | 91.61% | 90.63% | **91.12%** | 3,413,657 | 3 | 326.4s |
| **YOSM (Yoruba)** | TextCNN | 81.00% | 75.41% | 92.00% | **82.88%** | 2,931,801 | 8 | 10.3s |
| **YOSM (Yoruba)** | CNN-BiLSTM-Attention | 81.80% | 77.89% | 88.80% | **82.99%** | 3,523,657 | 5 | 21.5s |
