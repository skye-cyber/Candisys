# Model Training perfomance

## Decision Tree
 python train.py -m decision_tree
- INFO - Dataset: /home/skye/Candisys/datasets/open_dataset.csv
--------------------------------------------------
- INFO - Training Decision Tree...
--------------------------------------------------
Decision Tree's Accuracy is: 98.41%
--------------------------------------------------
Classification Report for Decision Tree:
--------------------------------------------------
              precision    recall  f1-score   support

       False       0.99      0.99      0.99    170525
        True       0.84      0.84      0.84      9014

    accuracy                           0.98    179539
   macro avg       0.92      0.92      0.92    179539
weighted avg       0.98      0.98      0.98    179539

--------------------------------------------------
 Cross-validation scores for Decision Tree: [0.98372498 0.98340184 0.98405352 0.98428745 0.98386971]
--------------------------------------------------
- INFO - Decision Tree model saved to /home/skye/Candisys/models/DecisionTree.pkl
--------------------------------------------------


└─$ python train.py -m decision_tree
- INFO - Dataset: /home/skye/Candisys/datasets/open_dataset.csv
--------------------------------------------------
- INFO - Training Decision Tree...
--------------------------------------------------
Decision Tree's Accuracy is: 99.65%
--------------------------------------------------
Classification Report for Decision Tree:
--------------------------------------------------
              precision    recall  f1-score   support

       False       1.00      0.99      1.00    170398
        True       0.99      1.00      1.00    170938

    accuracy                           1.00    341336
   macro avg       1.00      1.00      1.00    341336
weighted avg       1.00      1.00      1.00    341336

--------------------------------------------------
Cross-validation scores for Decision Tree: [0.99653128 0.99641702 0.99641116 0.99660452 0.99653128]


python train.py -m decision_tree
- INFO - Dataset: /home/skye/Candisys/datasets/open_dataset.csv
--------------------------------------------------
- INFO - Training Decision Tree...
--------------------------------------------------
Decision Tree's Accuracy is: 100.00%
--------------------------------------------------
Classification Report for Decision Tree:
--------------------------------------------------
              precision    recall  f1-score   support

       False       1.00      1.00      1.00    170738
        True       1.00      1.00      1.00    170786

    accuracy                           1.00    341524
   macro avg       1.00      1.00      1.00    341524
weighted avg       1.00      1.00      1.00    341524

--------------------------------------------------
Cross-validation scores for Decision Tree: [1. 1. 1. 1. 1.]
--------------------------------------------------
- INFO - Decision Tree model saved to /home/skye/Candisys/models/DecisionTree.pkl
--------------------------------------------------

## RFC-> Random Forest Classifier
 python train.py -m RFC          
- INFO - Dataset: /home/skye/Candisys/datasets/open_dataset.csv
--------------------------------------------------
- INFO - Training Random Forest Classifier...
--------------------------------------------------
Random Forest Classifier's Accuracy is: 100.00%
--------------------------------------------------
Classification Report for Random Forest Classifier:
--------------------------------------------------
              precision    recall  f1-score   support

       False       1.00      1.00      1.00    170738
        True       1.00      1.00      1.00    170786

    accuracy                           1.00    341524
   macro avg       1.00      1.00      1.00    341524
weighted avg       1.00      1.00      1.00    341524

--------------------------------------------------
Cross-validation scores for Random Forest Classifier: [1. 1. 1. 1. 1.]
--------------------------------------------------
- INFO - Random Forest Classifier model saved to /home/skye/Candisys/models/RFClassifier.pkl
--------------------------------------------------
