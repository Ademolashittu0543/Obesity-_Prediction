#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
df.head()

   Column Name               	           Meaning
-Gender	                            Male or Female
-Age	                            Age of the person
-Height	                            Height (in meters)
-Weight	                            Weight (in kg)
-family_history_with_overweight	    Yes/No
-FAVC	                            frequent high-calorie food (Yes/No)
-FCVC	                            Frequency of vegetable consumption
-NCP	                            Number of main meals per day
-CAEC	                            Consumption of food between meals
-SMOKE	                            Smoker (Yes/No)
-CH2O	                            Water consumption (liters/day)
-SCC	                            Calories monitor app user (Yes/No)
-FAF	                            Physical activity frequency
-TUE	                            Time spent using tech devices
-CALC	                            Alcohol consumption
-MTRANS	                            Mode of transportation
-NObeyesdad	                        Target variable (Obesity level)
# In[3]:


from pandas_extensions import ColumnSummaryAccessor
summary_df = df.column_summary.summary()
display(summary_df)


# In[4]:


df.isna().sum()


# In[5]:


df.shape


# In[6]:


df.info()


# Distribution of Obesity Levels

# In[8]:


plt.figure(figsize=(10,5))
sns.countplot(x='NObeyesdad', data=df)
plt.title('Distribution of Obesity Levels')
plt.xticks(rotation=45)
plt.show()

This chart shows the distribution of obesity levels among participants. It helps to identify class imbalance. Most participants fall under Obesity_Type_1
# Obesity Level Distribution by Gender

# In[10]:


sns.countplot(data=df, x='NObeyesdad', hue='Gender')
plt.title("Obesity Level Distribution by Gender")
plt.xticks(rotation=45)
plt.show()

              Obesity Level Distribution by Gender 
              
Key Observations
      
1. Obesity_Type_III:
    
• This category has the highest count overall.
    
• Females significantly outnumber males in this category.

2. Obesity_Type_II:
    
• Also a high-count category.
    
• Males have slightly fewer individuals than females, but both are high.

3. Obesity_Type_I and Overweight_Level_II:
    
• Males have higher counts than females.

4. Overweight_Level_I and Normal_Weight:
    
• Gender distribution is nearly equal in both categories.

5. Insufficient_Weight:
    
• More females fall into this category than males.

6. Overall Trend:
    
• Females are more represented in the extreme categoris ((Insufficient_Weight and Obesity_Type_III).
    
• Males dominate the mid-level overweight and obesity categorie  (Overweight_Level_II and Obesity_Type_I).ation between genders
# Age Distribution Across Obesity Levels

# In[12]:


plt.figure(figsize=(10,6))
sns.violinplot(x='NObeyesdad', y='Age', data=df)
plt.title('Age Distribution Across Obesity Levels')
plt.xticks(rotation=45)
plt.show()

               Age Distribution Across Obesity Levels

1. General Age Range:

    • Across all obesity categories, the majority of individuals are between ages 18 and 30.

    • Most categories have densely packed populations in the early 20s, especially the normal weight and overweight levels.


2. Insufficient Weight:

    • Individuals in this group are mostly younger, clustering tightly around ages 17–22.

    • The distribution is narrow, suggesting low age variability.


3. Obesity Type III:

    • This category has a tightly clustered age range, mostly between 20 and 25.

    • Despite being the most extreme obesity level, the age distribution is younger and less spread out.


4. Obesity Type II:

    • Shows a wider age spread, generally skewed toward older individuals, with a dense population around 30 years.


5. Obesity Type I and Overweight Levels I & II:

    • These categories have broader age ranges, with some individuals above 50.

    • This suggests people in these categories span a wide age group, potentially due to gradual weight gain over time.


6. Normal Weight:

    • Most individuals are in their early 20s, similar to the overweight groups, but with more age diversity.


Summary:

    • Younger individuals dominate the dataset across all categories.

    • Higher obesity levels (Type II & III) tend to occur in both younger and middle-aged adults.

    • Insufficient weight is primarily associated with teens and early adults, showing low variance.

    • As obesity severity increases, age distributions become more polarized, with fewer older individuals in extreme categories like Obesity Type III.
# Height vs Weight colored by Obesity Level

# In[14]:


plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Height', y='Weight', hue='NObeyesdad')
plt.title('Height vs Weight colored by Obesity Level')
plt.show()

                  Height vs Weight by Obesity Level
                  

1. Weight Progression Clusters:

    • Clear visual separation between obesity categories, showing gradual weight increase from Normal Weight to Obesity Type III.

    • Overweight groups (I-II) bridge the gap between normal and obese categories.

2. Height Matters:

    • Taller individuals (1.8m+) cluster in higher obesity categories at similar BMIs, suggesting height may influence risk thresholds.

    • Insufficient Weight appears across all heights, indicating non-height-related factors (e.g., metabolism, diet).

3. Critical Transition Zones:

    • 1.6–1.7m height range: Where weight distributions begin diverging sharply into obesity categories.

    • Outliers: Some normal-weight individuals at high weight/height (likely muscular builds).

4. Actionable Insight:

    • Taller individuals may need adjusted BMI thresholds for obesity screening.

    • Focus interventions on height-specific weight management (e.g., for those 1.6–1.8m).
# Critical Height-Weight Transition Zone (1.6m–1.8m)

# In[16]:


transition_zone = df[(df['Height'] >= 1.6) & (df['Height'] <= 1.8)]  
sns.lmplot(data=transition_zone, x='Height', y='Weight', hue='NObeyesdad', height=6)  
plt.title('Critical Height-Weight Transition Zone (1.6m–1.8m)')  

             Critical Transition Zone (1.6m–1.8m)
             
1. Weight Thresholds Vary by Heigt:h

> 1.6m to 1.7:

    • Normal Weight: 50–70kg

    • Obesity Type I begins: >100kg

> 1.7m–1.8m:

    • Normal Weight: 60–80kg

    • Obesity Type I begins: > 110k   10kg weight tolerance shift per 0.1m height increas )

2. High-Risk Clusters:

    • 1.65m–1.75m: Steepest obesity gradient (where most category transitions occur)

    • 1.7m: "Tipping point" where all obesity subtypes emerge

3. Outliers Tell a Story

    • Insufficient Weight: Flat distribution (40–60kg across all heights)

    • Obesity Type III: Only appears above 1.65m height
# Calculating Body Mass Index and compare across categories (Feature Engineering)

# In[18]:


df['BMI'] = df['Weight']/(df['Height']**2)
sns.boxplot(data=df, x='NObeyesdad', y='BMI', 
           order=['Insufficient_Weight','Normal_Weight',
                  'Overweight_Level_I','Overweight_Level_II',
                  'Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'])
plt.xticks(rotation=45)
plt.axhline(18.5, color='green', linestyle='--') 
plt.axhline(25, color='yellow', linestyle='--')   
plt.axhline(30, color='red', linestyle='--')     

                Key BMI Observations by Obesity Category

1. Clear BMI Progression:

    • Shows perfect stepwise increase from Insufficient Weight (lowest BMI) to Obesity Type III (highest BMI).

    • Each category represents a distinct BMI range with minimal overlap.

2. Critical Thresholds:

    • Normal Weight (center point) acts as the healthy baseline.

    • Overweight Levels I-II show the transition zone.

    • Obesity Types I-III demonstrate escalating severity.

3. Notable Patterns:

    • Largest BMI jump occurs between Overweight Level II and Obesity Type I.

    • Obesity Type III shows the widest BMI range (highest variability).

    • Insufficient Weight has the narrowest BMI distribution.

4. Clinical Implications:

    • Confirms BMI's effectiveness for classifying obesity levels.

    • Highlights need for different intervention strategies at each stage.

    • Suggests particular attention needed for Overweight-to-Obesity transition.
# In[19]:


bmi_ranges = df.groupby('NObeyesdad')['BMI'].agg(['min','median','max'])
print(bmi_ranges.sort_values('median'))


# Physical Activity vs Obesity Level

# In[21]:


plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='NObeyesdad', y='FAF')
plt.title('Physical Activity vs Obesity Level')
plt.xticks(rotation=45)
plt.show()

                  Physical Activity Patterns by Obesity Level

1. Inverse Relationship:

    • Normal Weight individuals show the highest physical activity levels.

    • Activity declines progressively across obesity categories, with Obesity Type III showing the lowest activity.

2. Critical Transition Points:

    • Biggest activity drop occurs between Overweight Level II → Obesity TypeI

    • Insufficient Weight group shows moderate activity, suggesting non-activity-related causes.

3. Notable Exceptions:

    • Some Obesity Type I individuals maintain moderate activity (possible metabolic factors).

    • Insufficient Weight group doesn't follow the trend (may indicate eating disorders or health conditions).
# Water Consumption by Obesity Level

# In[23]:


plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='NObeyesdad', y='CH2O')
plt.title('Water Consumption by Obesity Level')
plt.xticks(rotation=45)
plt.show()

                Key Water Consumption Insights by Obesity Category

1. Clear Hydration Gradient:

    • Normal Weight individuals show the highest water intake

    • Steady decline in consumption from Overweight Level I to Obesity Type III

    • Insufficient Weight group breaks the pattern (moderate intake despite low weight)

2. Critical Thresholds:

    • Biggest drop occurs between Overweight Level II → Obesity Type I

    • Obesity Type III shows lowest hydration levels (potential dehydration risk)

3. Clinical Implications:

    • Intervention Target: Focus on water intake for Overweight Level II individuals

    • Screening: Low hydration in Normal Weight may predict future weight gain

    • Research: Investigate why Insufficient Weight maintains moderate intake
# Alcohol Consumption Frequency vs Obesity

# In[25]:


plt.figure(figsize=(8,5))
sns.countplot(x='CALC', hue='NObeyesdad', data=df)
plt.title('Alcohol Consumption Frequency vs Obesity')
plt.xticks(rotation=45)
plt.show()

                  Alcohol Consumption Frequency vs Obesity

1. "Sometimes" is the most common alcohol consumption pattern:

• This category has the highest count across all obesity levels, especially for:

    • Obesity Type III (by far the most prominent)

    • Overweight Level I

    • Obesity Type II

2. "No" alcohol consumption:

• Obesity Type I has the highest number of non-drinkers, followed by Overweight Level II and Normal Weight.

• Interestingly, Insufficient Weight also has a relatively high number of non-drinkers.

3. "Frequently" drinking:

• This category has very low counts overall, with a slightly higher presence in:

    • Normal Weight

    • Overweight Level II

    • Obesity Type I

4. "Always" drinking:

• Extremely rare across all categories.

• Only a few individuals in Normal Weight and Obesity Type I engage in daily alcohol consumption.

5. Obesity Type III and alcohol:

• Most individuals in this category report drinking "Sometimes", with very few non-drinkers or frequent drinkers, suggesting moderate alcohol use may be associated with extreme obesity levels in this dataset.

6. Insufficient Weight group:

• Has a higher proportion of non-drinkers compared to drinkers.

• May suggest that individuals with insufficient weight are less likely to consume alcohol.



Summary:

• Moderate alcohol consumption ("Sometimes") is most associated with higher obesity levels, particularly Obesity Type II and III.

• Non-drinking is most common in Insufficient Weight and Obesity Type I.

• Frequent or constant drinking is rare, and not strongly associated with any particular weight class.
#  Transport Method (MTRANS) vs Obesity

# In[27]:


plt.figure(figsize=(9,5))
sns.countplot(data=df, x='MTRANS', hue='NObeyesdad')
plt.title("Mode of Transport vs Obesity Levels")
plt.xticks(rotation=45)
plt.show()

                    Mode of Transport vs Obesity Levels

1. Public Transportation Dominates:

• This is the most common mode of transport across all obesity levels.

• Obesity Type III has the highest number of public transport users, followed by Obesity Type I and Overweight Level I.

• Suggests public transport may be widely used regardless of weight, possibly due to accessibility or socioeconomic factors.

2. Automobile Usage:

• Noticeably higher in Obesity Type I and II individuals.

• May suggest a correlation between frequent automobile use and moderate obesity.

• Normal weight individuals use cars the least among the higher-obesity categories.

3. Walking:

• More common among those with Normal Weight and Insufficient Weight.

• Very low for higher obesity levels (especially Obesity Type III), indicating reduced physical activity could be a contributing factor.

4. Bike and Motorbike Use:

• Extremely rare across all categories.

• Slightly more common among Normal Weight individuals but still negligible overall.

• Could imply these forms of active or semi-active transport are underutilized.

5. Obesity Type III and Transport**:

• Almost entirely reliant on public transportation.

• Very little walking or automobile use.

• Suggests potential mobility issues or lifestyle constraints.



Summary:

• Public transportation is the dominant mode of travel for all, especially heavily obese individuals.

• Walking and cycling are more associated with lower obesity levels, while automobile reliance grows with obesity severity.

• Promoting active transport options like walking or cycling may help reduce obesity rates, especially in urban populations.
# In[28]:


df


# In[29]:


categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("Categorical variables:")
print(categorical_cols)

numeric_cols = df.select_dtypes(include=['int64', 'float64','int32','float32']).columns.tolist()

print("numeric variables:")
print(numeric_cols)


# Outlier Analysis Across Key Features

# In[30]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

Looking at the boxplots, here’s where outliers are present:

1. Age: Outliers are present on the higher end (above ~40 years)
   
    Possible Reason for Outliers: The dataset might be targeting a younger population (e.g., students or young adults), so older individuals (above 40 or 50) appear as outliers. These could be parents, instructors, or incorrectly entered data()

2. Height: A few outliers on the higher side (~above 1.9 meters).

    Possible Reason for Outliers: People significantly taller than average (e.g., above 1.9m) could naturally be outliers due to human height distribution. Alternatively, errors like mixing units (e.g., inches instead of meters) can cause this.

3. Weight: Clear outliers on the higher end (~above 150 kg).

    Possible Reason for Outliers: A few individuals may have obesity or health conditions that result in higher weights. Alternatively, incorrect data entries (like extra zeros) or mixing up pounds and kilograms can cause spikes.
   
4. NCP: Heavy presence of outliers at 1 and 4

    Possible Reason for Outliers: Outliers in this column may be due to incorrect data entry, such as the use of float values where only whole numbers are expected.
   
5. CH2O: No visible outliers.
    
6. FAF: No significant outliers.
    
7. TUE: No significant outliers.
    
8. FCVC: No visible outliers.

9. BMI: No visile outliers.
# In[31]:


df['Age']=df['Age'].round().astype(int)
df['NCP']=df['NCP'].round().astype(int)


# In[32]:


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['Age', 'Height', 'Weight', 'NCP']:
    df_cleaned = remove_outliers_iqr(df, col)

print("New dataset shape after outlier removal:", df_cleaned.shape)


# In[33]:


df_cleaned


# In[34]:


from sklearn.preprocessing import LabelEncoder

df_clean = df_cleaned.copy()

le = LabelEncoder()

for col in categorical_cols:
    df_clean[col] = le.fit_transform(df_clean[col])
    
df_clean


# In[35]:


df_clean.describe()


# In[36]:


df_clean.isna().sum()


# In[37]:


df_clean.info()


# Correlation Matrix

# In[39]:


plt.figure(figsize=(12,8))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

                       Correlation Matrix Analysis

1. Strongest Correlations

• Weight and BMI: Extremely strong positive correlation (0.94), which is expected since BMI is calculated from weight and height.

• Weight and Height: Moderate positive correlation (0.45), taller individuals tend to weigh more.

• Weight and Family History: Moderate correlation (0.48), suggesting genetic factors in weight.

• Gender and FCVC: Moderate negative correlation (-0.37), possibly indicating different eating habits between genders.

2. Moderate Correlations

• Age and MTRANS: Strong positive correlation (0.63), likely reflecting transportation choices changing with age.

• FAVC and Weight: Moderate correlation (0.30), frequent high-calorie food consumption relates to higher weight.

• CH2O and Weight: Moderate correlation (0.24), water intake relates to weight.

3. Weak Correlations

• Most other correlations are relatively weak (<0.3), including:

    • Smoking habits with other variables

    • Physical activity factors (FAF, TUE)

    • Most dietary habits (NCP, CAEC)

4. Notable Observations

• BMI shows the strongest relationships, particularly with weight (0.94) and family history (0.48).

• Gender correlates moderately with height (0.63), likely reflecting biological differences.

• Age has relatively weak correlations with most health factors except transportation (MTRANS).
# In[40]:


x=df_clean.drop(columns=['NObeyesdad'],axis=1)
y=df_clean.NObeyesdad


# In[41]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

pca = PCA()
x_pca=pca.fit_transform(X_scaled)


# PCA Explained Variance Ratio

# In[43]:


plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance Ratio')
plt.grid()
plt.show()

                             PCA Explained Variance Ratio

1. Dimensionality Reduction Insight:

• The first 10 principal components explain around 90% of the total variance in the data.

• This means you can reduce your dataset to 10 features without losing much information—ideal for simplifying models while retaining accuracy.

2. Elbow Point Observation:

• There’s a visible "elbow" around component 10–12, after which the gains in explained variance start to flatten out.

• This suggests diminishing returns beyond this point—adding more components contributes little additional information.

3. Optimal Component Selection:

• If you're aiming for a balance between simplicity and data retention:

• 7–10 components = Good tradeoff (~85–90% variance)

• 12+ components = Higher accuracy, but with more complexity

4. Feature Engineering Benefit:

• PCA has likely successfully identified and compressed correlated features, making downstream tasks like clustering, classification, or regression more efficient and interpretable.
# Geting feature importance from PCA components

# In[45]:


loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i}' for i in range(1, len(x.columns)+1)],
    index=x.columns
)

top_n_components = 5
print(f"\nTop features per principal component (PC1-{top_n_components}):")
print(loadings.iloc[:, :top_n_components].abs().idxmax())

feature_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': np.sum(np.abs(loadings.iloc[:, :top_n_components]), axis=1)
}).sort_values('Importance', ascending=False)

print("\nOverall feature importance from PCA:")
print(feature_importance.head(17))


# In[46]:


x_pca.shape


# In[47]:


from sklearn.model_selection import train_test_split
X_pca_train, X_pca_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_pca_train.shape, X_pca_test.shape, y_train.shape, y_test.shape


# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='accuracy', verbose=2)

grid_search.fit(X_pca_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy Score:", grid_search.best_score_)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_pca_test)


# In[49]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
class_names = le.classes_
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()

     Confusion matrix for a Random Forest classifier used for prediction 

1. High Accuracy: The matrix shows strong diagonal dominance, which means most predictions match the actual labels. For example:

• Insufficient Weight: 35 correctly classified.

• Obesity Type III: 58 correctly classified.

• Overweight Level II: 34 correctly classified.


2. Minimal Misclassifications:

• The only visible misclassification is 2 instances of Overweight Level I being incorrectly classified as Normal Weight.

• No other classes show noticeable confusion with one another.



3. Balanced Performance Across Classes:

• All classes, including various obesity types and weight levels, are well predicted.

• No particular class appears underrepresented or severely misclassified.


Conclusion:
The Random Forest model demonstrates excellent performance in classifying body weight categories with only a minor error rate. It's especially good at distinguishing between the different obesity and overweight levels, which can often be tricky in classification tasks.
# In[51]:


import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_pca_train, y_train)

explainer = shap.TreeExplainer(rf)

shap_values = explainer.shap_values(X_pca_test)

if isinstance(shap_values, list):
    mean_shap_values = np.abs(shap_values).mean(0)
else:
    mean_shap_values = shap_values

plt.figure(figsize=(10, 8))
shap.summary_plot(
    mean_shap_values, 
    X_pca_test,
    plot_type='bar',
    show=False
)
plt.title('Feature Importance via SHAP Values (Bar Plot)', fontsize=14)
plt.tight_layout()
plt.show()

              Feature Importance via SHAP Values (Bar Plot)

Based on the SHAP (SHapley Additive exPlanations) values bar plot shown, here are the key insights about feature importance in this model:

1. Top Influential Features

• BMI (Body Mass Index) is by far the most important feature in the model, with the highest mean absolute SHAP value (around 0.7).

• Weight is the second most important feature.

• FCVC (Frequency of consumption of vegetables) is third.

• Gender, Height, and Age follow as moderately important features.


2. Less Influential Features

• The bottom group of features have relatively low importance, including:

    • Family history with overweight

    • CALC (Consumption of alcohol)

    • CAEC (Consumption of food between meals)

    • TUE (Time using technology devices)

    • CH2O (Water consumption)

    • FAF (Physical activity frequency)

    • FAVC (Frequent consumption of high caloric food)

    • MTRANS (Transportation used)

    • SCC (Calories consumption monitoring)

    • SMOKE (Smoking status)

    • NCP (Number of main meals)


Insights

• The model appears to rely most heavily on direct body measurement metrics (BMI, weight, height) and basic demographic information (gender, age).

• Dietary and lifestyle factors have relatively minor impact on the model's predictions compared to the biometric measurements.

• The long tail of less important features suggests the model considers many factors but weights them much less than the top features.