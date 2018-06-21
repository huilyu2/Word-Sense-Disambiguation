## Data Preprocess
Due to the large volumn of the generated dataset, we didn't upload the dataset
to this repository.
The following scripts generate the dataset:

`parser.py` generates the dataset for multi-class classification, from SemCor
corpus.

`split_dataset.py` splits the dataset produced by parser into training set and
test set, in 70/30 ratio. The split is stratified to account for class imbalance.

## Learning Algorithms

`bayes.py`: Naive Bayes

`logistic_regression.ipynb`: Logistic Regression

`svm.ipynb`: Support Vector Machine

`waveNet_keras/train.py `: Neural Network
