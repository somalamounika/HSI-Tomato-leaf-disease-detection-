# 3-class stage model

Labels from the use-case (not inoculated vs control):
- **1dai → Healthy**
- **3dai → Early infection**
- **6dai → Infected**

Selected: **lda**
Leave-one-leaf-out balanced accuracy: 1.000
Accuracy: 1.000  macro-F1: 1.000

Leaves per class: early=8, healthy=8, infected=10

```
              precision    recall  f1-score   support

     healthy       1.00      1.00      1.00         8
       early       1.00      1.00      1.00         8
    infected       1.00      1.00      1.00        10

    accuracy                           1.00        26
   macro avg       1.00      1.00      1.00        26
weighted avg       1.00      1.00      1.00        26

```