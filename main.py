import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_students = 200
data = {
    'GPA': np.random.uniform(0.0, 4.0, num_students),
    'SocialMediaEngagement': np.random.randint(1, 11, num_students),  # Scale of 1-10
    'LibraryVisits': np.random.randint(0, 20, num_students),
    'Absences': np.random.randint(0, 20, num_students),
    'AtRisk': np.random.randint(0, 2, num_students) # 0: Not at risk, 1: At risk
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering (Minimal in this synthetic example) ---
# In a real-world scenario, this section would involve handling missing values, outliers, etc.
# For this example, the data is already clean.
# --- 3. Data Analysis and Model Building ---
X = df[['GPA', 'SocialMediaEngagement', 'LibraryVisits', 'Absences']]
y = df['AtRisk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
# --- 4. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Student Attributes')
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")
plt.figure(figsize=(8,6))
sns.scatterplot(x='GPA', y='SocialMediaEngagement', hue='AtRisk', data=df, palette=['blue', 'red'])
plt.title('GPA vs. Social Media Engagement')
plt.savefig('gpa_socialmedia.png')
print("Plot saved to gpa_socialmedia.png")
plt.figure(figsize=(8,6))
sns.boxplot(x='AtRisk', y='Absences', data=df, palette=['blue', 'red'])
plt.title('Absences by At-Risk Status')
plt.savefig('absences_risk.png')
print("Plot saved to absences_risk.png")