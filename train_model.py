import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# Set style for plots
sns.set(style="whitegrid")

def perform_eda(df):
    """Generate and save EDA plots."""
    os.makedirs('eda_plots', exist_ok=True)
    
    # 1. Survival Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Survived', data=df, palette='viridis')
    plt.title('Survival Distribution (0 = No, 1 = Yes)')
    plt.savefig('eda_plots/survival_dist.png')
    plt.close()

    # 2. Survival by Gender
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Sex', hue='Survived', data=df, palette='magma')
    plt.title('Survival by Gender')
    plt.savefig('eda_plots/survival_gender.png')
    plt.close()

    # 3. Survival by Passenger Class
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='coolwarm')
    plt.title('Survival by Passenger Class')
    plt.savefig('eda_plots/survival_class.png')
    plt.close()

    # 4. Age Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'].dropna(), bins=30, kde=True, color='blue')
    plt.title('Age Distribution')
    plt.savefig('eda_plots/age_dist.png')
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig('eda_plots/correlation_heatmap.png')
    plt.close()
    
    print("EDA plots saved in 'eda_plots/' directory.")

def train_model():
    # Load dataset
    train_path = 'Titanic_train.csv'
    if not os.path.exists(train_path):
        train_path = '/Users/ruturaj/Downloads/Logistic Regression/Titanic_train.csv'
        
    df = pd.read_csv(train_path)
    
    print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Perform EDA
    perform_eda(df)
    
    # Step 1: Dataset Understanding (Print Info)
    print("\nMissing Values:\n", df.isnull().sum())
    
    # Define features and target
    target = 'Survived'
    # Features mentioned in Step 8 for Streamlit App
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    X = df[features]
    y = df[target]
    
    # Preprocessing Pipeline
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Sex', 'Embarked']
    
    # Impute missing values then Scale/Encode
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('pclass', 'passthrough', ['Pclass'])  # Pclass is numeric but categorical in nature, keeping as is or we could encode it too.
        ])
    
    # Full Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model_pipeline.fit(X_train, y_train)
    print("\nModel trained successfully!")
    
    # Evaluation
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    print("\nModel Evaluation Metrics:")
    for m, val in metrics.items():
        print(f"{m}: {val:.4f}")
        
    # Visualization: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('eda_plots/confusion_matrix.png')
    plt.close()
    
    # Visualization: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC-AUC: {metrics["ROC-AUC"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('eda_plots/roc_curve.png')
    plt.close()
    
    # Interpretation
    classifier = model_pipeline.named_steps['classifier']
    preprocessor_obj = model_pipeline.named_steps['preprocessor']
    
    # Get feature names after transformation
    cat_features_names = preprocessor_obj.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_features_names) + ['Pclass']
    
    coefs = classifier.coef_[0]
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    
    print("\nModel Coefficients (Interpretation):")
    print(coef_df)
    
    # Save Model components
    joblib.dump(model_pipeline, 'model.pkl')
    # Or separate if needed, but saving the full pipeline is cleaner for Streamlit
    joblib.dump(preprocessor_obj, 'preprocessor.pkl')
    # Actually, saving the full pipeline is best for app.py
    
    print("\nModel components saved: 'model_full_pipeline.pkl' and 'preprocessor.pkl'")

if __name__ == "__main__":
    train_model()
