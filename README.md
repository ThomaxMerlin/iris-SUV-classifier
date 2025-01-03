# Iris Dataset Classification using SVM  
**With Jupyter Notebook**

This Jupyter Notebook demonstrates how to build a Support Vector Machine (SVM) model to classify the Iris dataset into three species: `setosa`, `versicolor`, and `virginica`. The dataset used is `iris.csv`, which contains 150 entries with features like `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, and `PetalWidthCm`.

---

## **Table of Contents**
1. [Prerequisites](#prerequisites)
2. [Getting Started](#getting-started)
3. [Running the Code](#running-the-code)
4. [Code Explanation](#code-explanation)
5. [Results](#results)
6. [License](#license)

---

## **Prerequisites**
Before running the code, ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  ```bash
  pip install numpy pandas seaborn scikit-learn jupyter
  ```
- Jupyter Notebook (to run the `.ipynb` file).

---

## **Getting Started**
1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/iris-SUV-classifier.git
   cd iris-SUV-classifier
   ```

2. **Download the Dataset**  
   Ensure the dataset `iris.csv` is in the same directory as the notebook.

3. **Launch Jupyter Notebook**  
   Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Open the `.ipynb` file from the Jupyter Notebook interface.

---

## **Running the Code**
1. Open the `.ipynb` file in Jupyter Notebook.
2. Run each cell sequentially to execute the code.

---

## **Code Explanation**
### **1. Import Libraries**
```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```
- Libraries used for data manipulation, visualization, and modeling.

### **2. Load and Explore Data**
```python
data = pd.read_csv("iris.csv")
data.head()
data.describe()
data.info()
```
- Load the dataset and explore its structure, summary statistics, and data types.

### **3. Data Preprocessing**
```python
data = data.drop(columns='Id')
```
- Drop the `Id` column as it is not relevant for classification.

### **4. Data Visualization**
```python
sns.pairplot(data, hue='Species')
```
- Visualize the relationships between features using a pairplot, colored by species.

### **5. Prepare Data for Modeling**
```python
x = data.drop(columns='Species')
y = data['Species']
```
- Separate the features (`x`) and target variable (`y`).

### **6. Train-Test Split**
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=0)
```
- Split the data into training and testing sets.

### **7. Build and Train Model**
```python
iris_classifier = SVC()
iris_classifier.fit(x_train, y_train)
```
- Train an SVM classifier on the training data.

### **8. Evaluate Model**
```python
predictions = iris_classifier.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
```
- Evaluate the model using accuracy score.

---

## **Results**
- **Accuracy**: The model achieves an accuracy of **91.67%** on the test set.
- **Visualization**: The pairplot provides insights into the relationships between features and species.
- **Predictions**: The model can classify iris flowers into the correct species based on input features.

---

## **License**
This project is open-source and available under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as needed.

---

## **Support**
If you encounter any issues or have questions, feel free to open an issue in this repository or contact me at [minthukywe2020@gmail.com](mailto:minthukywe2020@gmail.com).

---

Enjoy exploring the Iris classification model in Jupyter Notebook! 🚀
