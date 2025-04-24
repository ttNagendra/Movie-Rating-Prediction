# Movie Rating Prediction Model

A machine learning model to predict IMDb movie ratings for Indian films using ensemble learning and advanced feature engineering techniques.

## Dataset
The dataset (`IMDb Movies India.csv`) contains information about Indian movies including movie titles, release years, director information, genre classifications, and IMDb ratings.

## Project Structure
- `movie_rating_prediction.ipynb`: Main Jupyter notebook
- `IMDb Movies India.csv`: Dataset
- `best_movie_rating_model.joblib`: Saved model
- `feature_scaler.joblib`: Feature scaler
- `feature_names.joblib`: Feature names
- `requirements.txt`: Dependencies

## Features
- Year of release
- Director statistics (movies directed, average rating, experience)
- Genre information and complexity
- Temporal features (decade, trends)

## Models
- Random Forest Regressor
- Gradient Boosting Regressor
- Weighted Ensemble

## Usage
```python
import joblib
import pandas as pd

# Load model components
model = joblib.load('best_movie_rating_model.joblib')
scaler = joblib.load('feature_scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

# Prepare movie data
movie_data = {
    'Year': 2024,
    'Director_Movies': 5,
    'Director_Avg_Rating': 7.5,
    'Director_Experience': 10,
    'Genre_Count': 2,
    'Director_Rating_Std': 0.5,
    'Decade': 2020
}

# Make prediction
movie_df = pd.DataFrame([movie_data])
scaled_features = scaler.transform(movie_df)
predicted_rating = model.predict(scaled_features)[0]
print(f"Predicted Rating: {predicted_rating:.1f}/10")
```

## Setup
1. Clone repository and navigate to directory:
```powershell
git clone <repository-url>
cd "Movie Rating Prediction"
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```powershell
jupyter notebook
```

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
