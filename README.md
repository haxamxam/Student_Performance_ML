# Student Performance Prediction

Student retention is considered an important aspect in several many enrollment management systems across the world. It is vital for educational institutes as it affects their university rankings, school reputation and financial well being. It has become an utmost priority for decision makers to focus on student retention. An essential step towards student retention is to understand student performance and the underlying factors that affect it. Such an understanding is the basis for accurately predicting at-risk students and appropriately intervening to retain them.


## Validation

To estimate the performance of the prediction models a 10-fold cross-validation approach was used in this case study.


## Model Selection and Libraries

```python
from sklearn.neural_network import MLPClassifier # MLP Neural Network Classifier first model 
from sklearn.linear_model import LogisticRegression #Logistic Regression second model
from sklearn.svm import SVC # SVM third model
from sklearn.tree import DecisionTreeClassifier # Decision tree Classifier fourth model
from sklearn import model_selection # mode selection for kfold cv
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#define all of our models
models = [
          ('Log_Regression', LogisticRegression(solver='liblinear')), 
          ('Dec_Tree', DecisionTreeClassifier(max_depth=5)),
          ('MLP_ANN', MLPClassifier(hidden_layer_sizes=(256,128,128, 32),activation="relu",random_state=1)),
          ('SVM', SVC(kernel = 'linear', gamma='scale'))
        ]


```

## Training and Testing Scores

<p align="center">
  <img src="https://github.com/haxamxam/student_performance/blob/main/student_performance.png" width="500" title="train">
  <img src="https://github.com/haxamxam/student_performance/blob/main/student_performance_1.png" width="500" alt="test">
</p>

## Confusion Matrix

<p align="center">
  <img src="https://github.com/haxamxam/student_performance/blob/main/confusion.png" width="500" title="train">
  <img src="https://github.com/haxamxam/student_performance/blob/main/confusion_1.png" width="500" alt="test">
</p>
