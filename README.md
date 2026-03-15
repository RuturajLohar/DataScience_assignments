# Titanic Survival Prediction Project

This project implements an end-to-end Machine Learning pipeline to predict survival on the Titanic using Logistic Regression.

## Project Structure
```
titanic_streamlit_project/
│
├── app.py              # Streamlit Web Application
├── train_model.py      # Data Analysis, EDA, and Model Training Script
├── Titanic_train.csv   # Training Dataset
├── Titanic_test.csv    # Test Dataset
├── model.pkl           # Saved Model Pipeline (incl. Preprocessor)
├── preprocessor.pkl    # Saved Preprocessor (Separate component)
├── requirements.txt    # Project Dependencies
├── README.md           # Deployment Instructions
└── .gitignore          # Git ignore file
```

## Features Used
- **Pclass**: Passenger Class
- **Sex**: Passenger Gender
- **Age**: Passenger Age
- **SibSp**: Number of Siblings/Spouses aboard
- **Parch**: Number of Parents/Children aboard
- **Fare**: Passenger Fare
- **Embarked**: Port of Embarkation (S, C, Q)

## Local Setup
1. Clone the repository or download the project files.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the training script (optional, as `model.pkl` is provided):
   ```bash
   python train_model.py
   ```
5. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Cloud Deployment (Streamlit Community Cloud)
1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Titanic Survival Predictor"
   # Create a repo on GitHub and link it
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Click "New app".
   - Select the repository, branch (main), and the main file path (`app.py`).
   - Click "Deploy".

## Model Interpretation
- **Gender**: Female passengers had a significantly higher survival probability.
- **Class**: Higher class passengers (Lower `Pclass` value) were more likely to survive.
- **Age**: Younger passengers generally had higher survival rates in this model.
- **Fare**: Passengers who paid higher fares showed a slight increase in survival probability.
