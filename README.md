
## TV Show \& Movie Genre Prediction
## 📌 Project Overview
This project focuses on building machine learning models to predict the genre of TV shows and movies using metadata such as description, cast, director, and other attributes.

The goal is to demonstrate an end-to-end machine learning workflow including:
- Data Understanding
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Model Explainability

---

## 📂 Project Structure

tv-show-ml-project/

├── data/  
│   └── tv-shows.csv  
├── notebooks/  
│   └── tv_show_analysis.ipynb  
├── src/  
│   ├── data_loader.py  
│   ├── preprocess.py  
│   ├── feature_engineering.py  
│   ├── visualization.py  
│   ├── train.py  
│   ├── evaluate.py  
├── requirements.txt  
├── README.md  
├── report.docx


## ▶️ How to Run the Project
1. Open Anaconda Prompt
2. Navigate to project folder:
```bash
cd Downloads\tv-show-ml-project
```
3. Start Jupyter Notebook:
jupyter notebook
4. Open:
tv_show_analysis.ipynb
5. Run all cells sequentially.

---

## 📊 Dataset Information

The dataset contains metadata about TV shows and movies including:

- title
- director
- cast
- country
- release_year
- rating
- description
- platform
- listed_in (Genre)

### Target Variable:

**listed_in → Genre of the TV show or movie.**

This is a multi-class classification problem with multiple genre categories.

---

## 🧠 Feature Engineering

The following preprocessing steps were applied:

- Handling missing values using mean and mode
- Dropping unnecessary columns
- Combining text features
- TF-IDF vectorization
- Train-test split

### TF-IDF Parameters:

- max_features = 40000
- ngram_range = (1,3)
- stop_words = english

---

## 🤖 Models Used

Two machine learning models were trained:

### Model 1 — Logistic Regression
Reason:
- Works well with TF-IDF text features
- Efficient baseline model
---

### Model 2 — Linear Support Vector Machine (SVM)
Reason:
- Highly effective for text classification
- Handles large sparse feature spaces well
---

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Observed Performance:

- Logistic Regression Accuracy: ~61%
- Linear SVM Accuracy: ~62%

Best Model:

Linear SVM

---

## 📊 Visualizations Included

The following visualizations were created:

- Genre Distribution Plot
- Movie vs TV Show Distribution
- Release Year Distribution
- Target Class Distribution
- Missing Value Heatmap
- Confusion Matrix

---

## 📌 Conclusion

The Linear SVM model provided better performance for predicting genres.

The project demonstrates a complete machine learning pipeline including EDA, preprocessing, model training, evaluation, and explainability.

---

## 👤 Author

Name: K Charan Sri Teja

Project: TV Show & Movie Genre Prediction


## ⚙️ Installation

Install required libraries using:

```bash
pip install -r requirements.txt



