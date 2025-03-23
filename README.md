# Titanic Survival Prediction

## Project Overview
This project analyzes passenger survival patterns from the Titanic disaster using machine learning techniques. Through feature engineering and algorithmic comparison, the project identifies key factors determining survival probabilities and builds predictive models to classify whether a passenger survived or not.

## Dataset
The analysis uses the famous Titanic dataset containing information about passengers including:
- **Demographics:** Age, sex, passenger class
- **Ticket Information:** Fare, cabin, embarkation point
- **Family Relationships:** Number of siblings/spouses, parents/children onboard
- **Titles & Extracted Features:** Additional insights derived from existing data

## Features and Importance
The analysis reveals the following key survival factors:
- **Title_Mr (0.31822):** Being a male passenger was the most negative predictor of survival
- **Sex_female (0.19304):** Female passengers had significantly higher survival rates
- **Pclass_3 (0.08976):** Third-class passengers had lower survival probabilities
- **FamilySize (0.07400):** Family size influenced survival in a non-linear pattern
- **Title_Rare (0.04940):** Passengers with uncommon titles showed distinct survival patterns

## Models Evaluated
Several machine learning models were tested and evaluated:

| Model               | Accuracy | Precision | Recall  | F1 Score |
|---------------------|----------|-----------|---------|----------|
| LogisticRegression | 81.56%   | 0.7808    | 0.7703  | 0.7755   |
| RandomForest       | 82.68%   | 0.7867    | 0.7973  | 0.7919   |
| GradientBoosting   | 82.68%   | 0.8209    | 0.7432  | 0.7801   |
| SVC                | 81.01%   | 0.8030    | 0.7162  | 0.7571   |
| XGBoost            | **84.36%** | **0.8286** | **0.7838** | **0.8056** |

### Key Findings
- **XGBoost achieved the highest accuracy at 84.36%.**
- **Gender and social status were the strongest predictors of survival.**
- **Family dynamics showed a non-linear relationship with survival rates.**
- The analysis confirmed historical accounts of "women and children first" while revealing complex interactions between demographic factors.

## Repository Structure
```
├── data/
│   ├── train.csv          # Training dataset
│   └── test.csv           # Test dataset
├── notebooks/
│   ├── exploratory_data_analysis.ipynb  # Initial data exploration
│   ├── feature_engineering.ipynb        # Feature creation and transformation
│   └── model_comparison.ipynb           # Model training and evaluation
├── src/
│   ├── data_preprocessing.py            # Data cleaning and preparation
│   ├── feature_extraction.py            # Feature engineering functions
│   └── model_evaluation.py              # Model training and testing
├── images/
│   └── model_comparison.png             # Performance visualization
├── requirements.txt                     # Dependencies
└── README.md
```

## Setup and Installation

1. **Clone this repository:**
```sh
   git clone https://github.com/yourusername/titanic-survival-prediction.git
```

2. **Navigate to the project directory:**
```sh
   cd titanic-survival-prediction
```

3. **Install required dependencies:**
```sh
   pip install -r requirements.txt
```

4. **Run the Jupyter notebooks in order or execute the main script:**
```sh
   python src/main.py
```

## Future Improvements
- Implement ensemble methods combining the strengths of multiple models.
- Further feature engineering to extract additional insights.
- Hyperparameter optimization for the best-performing models.
- Survival probability analysis for different passenger profiles.

## License
This project is licensed under the **MIT License** - see the `LICENSE` file for details.

## Acknowledgments
- **Dataset:** Provided by Kaggle
- **Inspired by:** The historical significance of the Titanic disaster
- **Analysis Techniques:** Drawn from various machine learning resources

