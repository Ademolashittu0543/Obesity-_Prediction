 # Obesity Prediction Using Machine Learning
 
This project presents a supervised machine learning solution for predicting obesity levels in individuals based on their demographic data, lifestyle behaviors, and physiological measurements. The goal is to classify individuals into categories such as Normal Weight, Overweight, and various levels of Obesity using clean, interpretable models.

> ## 📊 Dataset

Source: UCI Machine Learning Repository – Obesity Levels Dataset

Size: 2,111 samples, 17 original features

Target Variable: NObeyesdad – multi-class label for weight status

> ## ⚙️ Methods & Workflow

➡️Exploratory Data Analysis (EDA) 

➡️Visualizations to explore weight, height, alcohol use, and transportation patterns

➡️Correlation heatmap and scatterplots for pattern discovery

➡️Feature Engineering

➡️Label encoding for categorical variables

➡️Standardization of numerical features

➡️PCA for dimensionality insight

➡️Modeling

➡️Classification using Random Forest

➡️Hyperparameter tuning with GridSearchCV

➡️Model evaluation via accuracy, precision, recall, f1-score, and confusion matrix

➡️Model Interpretability

➡️SHAP (SHapley Additive Explanations) used to interpret feature importance

> ## 💯 Results

✅Accuracy: ~99% on test set

✅Top Predictive Features: Weight, Height, FAF, Age

✅Interpretability: SHAP showed strong alignment between model decisions and medical intuition

> ## 📁 Files

Obesity dataset analysis.ipynb – main notebook

Report.pdf – academic report (optional if included)

Obesity dataset analysis.py - python code

README.md – this file

> ## 🧰 Tools & Libraries

🛠️Python (Pandas, Seaborn, Scikit-learn, SHAP, Matplotlib)

🛠️Jupyter Notebook

> ## 📌 Key Takeaways

This project demonstrates the power of combining machine learning with public health data to predict and understand obesity. It showcases the end-to-end pipeline from data cleaning to interpretability — and emphasizes how thoughtful modeling can support early diagnosis and intervention planning.
