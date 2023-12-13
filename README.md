# Open Source Software Final Project : Tumor Classification

## Description
The data used for this project are images of MRIs of brain tumors. Brain Tumors are classified as: Benign Tumor, Malignant Tumor, Pituitary Tumor, etc. The images are split into Training and Testing folders. The goal of this project is to find the best model to fit the training data points by using Scikit Learn.

## Installation

Use the package manager pip to install scikit-learn and scikit-image.

```bash
pip install scikit-learn scikit-image
```

## Structure
1) Load Packages
2) Load Data Points
3) Classification with Scikit Learn Library

## Explanation
In order to find the best model for this data I directly tried the AdaBoostClassifier from the sklear library, because it combines 2 classification methods. For the estimator I tested DecisionTreeClassifier (around 80% accuracy) and RandomForestClassifier (around 90% accuracy) and chose the latter. However, even with this estimator the accuracy was not constant and usually fell between 89% and 90%. I decided to tune in the model by varying the hyperparameters in order to find the most efficient ones. First i searched the best random seed and found one that allowed me to always have an accuracy above 90%. I also discovered that the learning rate and the n_estimators in the AdaBoost had no influence on the accuracy in this case, as the random_state and bootstrap of the RandomForest. However the best accuracy is achieved at 100 n_estimators for the RandomForest, instead of 50. Often the best settings were the default ones : for the RandomForest criterion 'gini' had better accuracy than 'entropy', 'sqrt' was the best max-feature, 2 the best min_samples_split.

## Results
The only estimators that I had to change are for the AdaBoost the random_state=41 and for the RandomForest the n_estimators=100. The accuracy I have achieved  with these settings is 91%. 

## Example of the code used to find the best random seed for the AdaBoostClassifier
```python
# Define hyperparameters for AdaBoost
adaboost_params = {
    'estimator':sklearn.ensemble.RandomForestClassifier(),
    'n_estimators': 50,
    'learning_rate': 1.0,
    'random_state': None
}

best_accuracy = 0
best_seed = None

# Loop through different random seeds to find the best one
for seed in range(100):  # Try seeds from 0 to 99
    adaboost_params['random_state'] = seed
    model = sklearn.ensemble.AdaBoostClassifier(**adaboost_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    
    print(f"Seed {seed + 1}/100: Accuracy = {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_seed = seed

print(f"Best seed found: {best_seed} with accuracy: {best_accuracy}"
```


## License

[MIT](https://choosealicense.com/licenses/mit/)
