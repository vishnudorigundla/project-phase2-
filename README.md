# project-phase2

## Title of the Project

EPILEPTIC SEIZURES RECOGNIZATION 

## About

The primary aim of this project is to develop a deep learning-based system for the accurate detection and classification of epileptic seizures using EEG signals. The project utilizes machine learning and deep learning techniques to analyze EEG data, identify seizure patterns, and distinguish between epileptic and non-epileptic events. This system aims to assist medical professionals in diagnosing epilepsy more efficiently, reducing manual effort, and improving the accuracy of seizure detection. Additionally, the project incorporates techniques such as Principal Component Analysis (PCA) for dimensionality reduction and a Stacking Classifier for enhanced classification performance.

## Features
1. Implements deep learning models such as Convolutional Neural Networks (CNNs), Deep Neural Networks (DNNs), and Random Forest for accurate seizure classification.
2. Utilizes Principal Component Analysis (PCA) to reduce high-dimensional EEG data while preserving essential information for better model efficiency.
3. Stacking Classifier combining multiple models, including Gradient Boosting and Support Vector Machines (SVM), to improve classification accuracy.
4. Real-time seizure detection capability, allowing faster identification and response.
5. High scalability, enabling the system to process large EEG datasets for extensive medical applications.
6. Optimized performance with techniques such as batch normalization and dropout to prevent overfitting and improve model generalization.


## Requirements
### Hard ware requirements:
```
●	Processor	:Multi-core CPU (Intel i5/i7 or AMD Ryzen 5/7)
●	Storage         : SSD (at least 256 GB)
●	GPU             : Optional (NVIDIA GTX 1660 or RTX 2060 for deep learning)
●	RAM             : Minimum 8 GB (16 GB preferred)
●	Keyboard        :110 keys enhanced
```
### Software requirements:
```
•	Operating System                   : Windows, Linux (Ubuntu), or macOS
•	 Programming Language              : Python
•	Machine Learning Libraries         : Scikit-learn, PCA,DNN, Pandas, NumPy, Matplotlib, Seaborn
•	 Development Environment           : Jupyter Notebook, Anaconda, PyCharm, or VS Code
•	 Vesion control                    : Git,Git hub
```
# System Architecture

![image](https://github.com/user-attachments/assets/0a6c06f8-ca3d-4faf-8a93-d82f6c27e559)

The architecture of this system processes EEG signal recordings through data preprocessing and normalization before classification using models like PCA, DNN, and Random Forest to detect epileptic or non-epileptic seizures. The classification results are further evaluated using metrics such as accuracy, precision, recall, and F1-score, with parameter tuning enhancing model performance.
# Program :
```
# Importing necessary packages

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
# Load dataset
data = pd.read_csv('/mnt/data/Epileptic Seizure Recognition.csv')

# Rename target column if necessary
if 'y' in data.columns:
    data.rename(columns={'y': 'target'}, inplace=True)
# Handle missing values
data.fillna(data.mean(), inplace=True)
# Separate features and target
X = data.drop(columns=['target'])
y = data['target']

# Convert all features to numeric
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(), inplace=True)
# Encode target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)




from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
# Train Gradient Boosting Model
gb_model = GradientBoostingClassifier()
gb_params = {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3]}
gb_grid = GridSearchCV(gb_model, gb_params, cv=2, n_jobs=-1, scoring='accuracy')
gb_grid.fit(X_train, y_train)
# Best Model
gb_best = gb_grid.best_estimator_
# Evaluate on test set
y_pred_gb = gb_best.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
# Create DNN Model
def create_dnn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
# Train Model
dnn_model = create_dnn_model(X_train.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = dnn_model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=64, callbacks=[early_stopping], verbose=1)
# Evaluate Model
dnn_loss, dnn_accuracy = dnn_model.evaluate(X_test, y_test)
print(f"DNN Test Accuracy: {dnn_accuracy}")


```
## Output

## Output1 - Head and Tail values for Train and test data:

![image](https://github.com/user-attachments/assets/8e3f38ea-4f7b-451a-b2c6-03afa302eb53)

![image](https://github.com/user-attachments/assets/0e339072-612f-4fde-8ffd-f3a74a9ea1d2)

## Output2 - PCA Explained variable for compression:

![image](https://github.com/user-attachments/assets/977a976a-57af-40ca-8b85-48e5c836a680)


## Output3 - classification report:

![image](https://github.com/user-attachments/assets/ebca8f25-b400-4826-951c-ae3fd0ae64dc)

## Output4 - DNN evaluation on test data:

![image](https://github.com/user-attachments/assets/50f73459-7434-4d20-b13c-c93f3805c5c3)


## Output5 -  DNN accuracy over Epochs:

![image](https://github.com/user-attachments/assets/01200161-1785-4ca9-8b17-72645e7f2182)

![image](https://github.com/user-attachments/assets/768a3848-73ca-45e5-845f-13cc4f017973)

## Output6 -  classification report for stacking classifier & Summary of models:

![image](https://github.com/user-attachments/assets/8edcc264-b54c-4e3a-9011-c6cea58db11c)

![image](https://github.com/user-attachments/assets/43a84827-5531-4395-8a0e-14e3082322ab)

## Results and Impact

The epileptic seizure detection system using deep learning provides an advanced and reliable method for classifying EEG signals into epileptic and non-epileptic categories. By leveraging machine learning techniques such as Gradient Boosting Classifier (GBC), Deep Neural Networks (DNN), Principal Component Analysis (PCA), and Stacking Classifiers, the system effectively analyzes EEG data and enhances diagnostic accuracy. The model's ability to extract meaningful features from EEG recordings ensures that seizure predictions are both precise and efficient.
### Future Work :

Furthermore, as part of my extended vision, The epileptic seizure detection system can be further improved by optimizing the deep learning models and enhancing interpretability for better medical applications. One of the key areas for enhancement is Deep Learning Model Optimization. The current system primarily uses Gradient Boosting Classifier (GBC), Deep Neural Networks (DNN), and Stacking Classifiers. However, future improvements can explore more advanced architectures such as Convolutional Neural Networks (CNNs) and Transformers. CNNs can effectively analyze EEG signals by capturing spatial patterns, while Transformers can leverage attention mechanisms to process long-range dependencies in EEG data. These improvements will enhance feature extraction, leading to better classification accuracy. Additionally, hyperparameter tuning techniques such as Bayesian Optimization can be employed to further refine model performance and reduce overfitting, ensuring better generalization across different datasets.

## Articles published / References 
```

[1]	Krishnan, M., Das, S., & Chakraborty, C. (2021). "Epileptic seizure detection using deep learning models with EEG signal analysis." Biomedical Signal Processing and Control, 68, 102690.
[2]	Rashid, R., et al. (2018). "Epileptic Seizure Detection Using Machine Learning: A Survey." International Journal of Computer Applications, 179(13), 21-25.
[3]	Zhang, Z., et al. (2019). "A Hybrid Approach for Epileptic Seizure Detection Using Convolutional Neural Networks and Support Vector Machines." IEEE Access, 7, 56775-56785.
[4]	Lee, S., et al. (2020). "Automatic Seizure Detection in Epileptic Patients Using Deep Learning Algorithms on EEG Data." Neural Computing and Applications, 32(16), 11867-11874.
[5]	Zhang, T., Yin, Z., Wang, W., & Zhao, J. (2019). "Automated epileptic seizure detection in EEG signals using deep neural networks." IEEE Transactions on Neural Systems and Rehabilitation Engineering, 27(2), 218-226.
[6]	 Hussein, R., Palangi, H., Ward, R. K., & Wang, Z. J. (2019). "Epileptic seizure detection: A deep learning approach using convolutional neural networks." Journal of Biomedical and Health Informatics, 23(4), 1767-1776.
[7]	Tsiouris, K. M., Pezoulas, V. C., Zervakis, M., Konitsiotis, S., Koutsouris, D. D., & Fotiadis, D. I. (2018). "A deep learning scheme for automated seizure detection in EEG signals using convolutional and recurrent networks." Journal of Biomedical Informatics, 84, 192-203.
[8]	Roy, Y., Banville, H., Albuquerque, I., Gramfort, A., Falk, T. H., & Faubert, J. (2019). "Deep learning-based electroencephalography analysis: A systematic review." Journal of Neural Engineering, 16(5), 051001.
[9]	Daoud, H., & Bayoumi, M. (2019). "Efficient epileptic seizure prediction based on deep learning." IEEE Transactions on Biomedical Circuits and Systems, 13(5), 804-813.
[10]	Ullah, I., Anwar, S. M., & Rehman, A. U. (2020). "Seizure detection in EEG signals using deep learning model." IEEE Access, 8, 166704-166712.
[11]	Golmohammadi, M., Ziyabari, S., Shah, V., Obeid, I., Picone, J., & Harrer, S. (2019). "Deep learning for Epileptic Seizure Detection: A Comparative Study." Journal of Neural Engineering, 16(3), 035002.
[12]	Birjandtalab, J., Pouyan, M. B., & Nourani, M. (2017). "Automated EEG-based epileptic seizure detection using deep learning approaches." IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(12), 2280-2287.
[13]	 Wang, J., Yu, H., Gao, J., & Wang, Y. (2021). "Epileptic seizure detection using CNN and improved multi-scale feature fusion." Neural Computing and Applications, 33, 12831-12845.
[14]	Thomas, J., Dasgupta, S., Ramesh, A., & Krishnan, P. (2018). "Deep learning in epileptic seizure detection: A benchmark study using EEG signals." Proceedings of the IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 1536-1541.
[15]	Chaovalitwongse, W. A., & Pardalos, P. M. (2018). "Computational neuroscience and epilepsy: Methods for seizure detection and treatment optimization." Springer Computational Neuroscience Series.
[16]	Khan, H., Marcuse, L., Fields, M., Swann, K., & Yener, B. (2017). "Focal onset seizure detection in EEG using deep learning." IEEE Transactions on Biomedical Engineering, 64(9), 2096-2105.
[17]	Rahman, M. M., & Bhuiyan, M. I. H. (2020). "Epileptic seizure classification using deep recurrent neural networks." IEEE Transactions on Biomedical Engineering, 67(6), 1595-1603.
[18]	Zhang, Y., Wang, X., Li, Z., & Li, X. (2021). "Hybrid deep learning for epileptic seizure classification." IEEE Access, 9, 37426-37437.




```


