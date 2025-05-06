# Insurance Premium Prediction

## Project Overview

The goal of this project is to predict insurance premiums based on various factors. This project predicts insurance premium amounts using advanced machine learning techniques, with a focus on robust feature engineering, data preprocessing, and model validation. The solution leverages LightGBM with cross-validation and ensemble methods to maximize predictive performance. The workflow is designed for reproducibility and extensibility.



## Directory Structure

```
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── insurance_premium_prediction.py
├── insurance_premium_prediction.ipynb
├── predictions.csv
├── feature_importance.png
├── requirements.txt
```

- **data/**: Contains the training, test, and sample submission datasets.
- **insurance_premium_prediction.py**: Main script for data processing, feature engineering, model training, and prediction.
- **insurance_premium_prediction.ipynb**: Jupyter notebook version for interactive exploration.
- **predictions.csv**: Output file with predicted insurance premiums.
- **feature_importance.png**: Visualization of feature importances.
- **requirements.txt**: List of required Python packages.



## Data Description

- **train.csv**: Training data with features and target variable (`Premium Amount`).
- **test.csv**: Test data for which predictions are to be made.
- **sample_submission.csv**: Example format for the submission file.

Key features include:
- Demographics (Age, Number of Dependents, Gender, Marital Status, Education Level)
- Financial (Annual Income, Credit Score)
- Health (Health Score)
- Policy details (Policy Start Date, Insurance Duration, Previous Claims, Vehicle Age)
- Lifestyle (Customer Feedback, Smoking Status, Exercise Frequency, Property Type)



## Workflow

1. **Data Loading & Exploration**
   - Loads training and test datasets.
   - Displays sample data and checks for missing values.

2. **Feature Engineering**
   - Extracts rich date features from policy start date (year, month, day, weekday, quarter, etc.).
   - Applies log, square, cube, and binning transformations to key numeric features.
   - Creates interaction and ratio features for deeper insights.

3. **Preprocessing**
   - Handles missing values (median for numerics, mode for categoricals).
   - Applies Yeo-Johnson power transformation for normalization.
   - One-hot encodes categorical variables.
   - Ensures train and test sets have matching columns.

4. **Model Training & Validation**
   - Uses LightGBM with 3-fold cross-validation (KFold).
   - Trains multiple models and averages their predictions (ensemble).
   - Evaluates using RMSLE (Root Mean Squared Logarithmic Error).

5. **Prediction & Output**
   - Makes predictions on the test set.
   - Averages predictions from ensemble and final model.
   - Outputs results to `predictions.csv`.

6. **Feature Importance**
   - Calculates and displays the most important features.
   - Saves a feature importance plot (`feature_importance.png`).



## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- lightgbm

Install dependencies with:
```bash
pip install -r requirements.txt
```



## Usage

1. **Prepare Data**
   - Place `train.csv` and `test.csv` in the `data/` directory.

2. **Run the Script**
   ```bash
   python insurance_premium_prediction.py
   ```

3. **Output**
   - Predictions will be saved to `predictions.csv`.
   - Feature importance plot will be saved as `feature_importance.png`.



## Results

- The model was evaluated using Root Mean Squared Logarithmic Error (RMSLE).
- **Final Mean CV RMSLE:** 1.0524 (+/- 0.0012)
- The script prints cross-validation RMSLE scores and the top 10 most important features.
- The output file `predictions.csv` contains two columns: `id` and `premium`.



## Acknowledgements

- The dataset used in this project was taken from Kaggle.
- This project is based on the Kaggle competition: [Playground Series - Season 4, Episode 12](https://www.kaggle.com/competitions/playground-series-s4e12)



## Notes

- The script uses log transformation on the target variable for better model performance.
- Extensive feature engineering is performed for improved accuracy.
- The project is designed for reproducibility and can be extended with additional models or features. 
