import os
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from logger import logging
from exception import CustomException
from utils import evaluate_model, save_obj
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Starting model training process")
            
            
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]  # Features and target for training
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]      # Features and target for testing


            # Define models
            models = {
                'LogisticRegression' : LogisticRegression(),
                'DecisionTreeClassifier' : DecisionTreeClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'GradientBoostingClassifier' : GradientBoostingClassifier(),
                'KNeighborsClassifier' : KNeighborsClassifier(),
                'Support Vector Classifier' : SVC(),
                'CatBoostClassifier' : CatBoostClassifier(task_type='GPU', devices='0'),
                'XGBClassifier' : XGBClassifier(tree_method='gpu_hist',gpu_id= 0,  max_depth=6, max_bin=256)
            }

            # Define hyperparameters for each model
            params = {
                # "LogisticRegression": {
                #     'penalty': ['l1', 'l2', 'elasticnet', None],  # Regularization type
                #     'C': [0.01, 0.1, 1, 10],                    # Inverse of regularization strength
                #     'solver': ['liblinear', 'saga'],             # Optimization algorithms
                #     'max_iter': [500, 1000, 1500, 2000]                  # Maximum iterations
                # },
                "LogisticRegression": {
                    'C': [0.01], 'max_iter': [500], 'penalty': ['l2'], 'solver': ['saga']             # Maximum iterations
                },
                # "DecisionTreeClassifier": {
                #     'criterion': ['gini', 'entropy', 'log_loss'],  # Split quality criterion
                #     'splitter': ['best', 'random'],               # Split strategy
                #     'max_depth': [None, 10, 20, 30, 50],          # Maximum tree depth
                #     'min_samples_split': [2, 5, 10],              # Minimum samples for a split
                #     'min_samples_leaf': [1, 2, 4],                # Minimum samples per leaf
                #     'max_features': [None, 'sqrt', 'log2'],       # Number of features to consider
                # },
                "DecisionTreeClassifier": {
                    'criterion': ['entropy'], 
                    'max_depth': [10], 
                    'max_features': ['log2'], 
                    'min_samples_leaf':[ 4], 
                    'min_samples_split': [2], 
                    'splitter': ['random']      # Number of features to consider
                },
                "RandomForestClassifier": {
                    'n_estimators': [100, 200],         # Number of trees
                    'max_depth': [None, 10, 20, 30, 50],           # Maximum tree depth
                    'min_samples_split': [2, 5, 10],               # Minimum samples for a split
                    'min_samples_leaf': [1, 2, 4],                 # Minimum samples per leaf
                    'max_features': ['sqrt', 'log2'],        # Features per split
                    'bootstrap': [True, False],                    # Bootstrapping strategy
                },
                "GradientBoostingClassifier": {
                    'loss': ['log_loss', 'deviance', 'exponential'],  # Loss function
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],          # Step size
                    'n_estimators': [100, 200, 500],                  # Boosting stages
                    'max_depth': [3, 5, 7],                           # Maximum depth of individual estimators
                    'min_samples_split': [2, 5, 10],                  # Minimum samples for a split
                    'max_features': ['auto', 'sqrt', 'log2'],         # Features for splitting
                    'subsample': [0.8, 0.9, 1.0],                     # Sampling fraction
                },
                "KNeighborsClassifier": {
                    'n_neighbors': [3, 5, 7, 9, 11],                 # Number of neighbors
                    'weights': ['uniform', 'distance'],              # Prediction weighting
                    'metric': ['euclidean', 'manhattan', 'minkowski'], # Distance metric
                },
                "SVC": {
                    'C': [0.1, 1, 10, 100],                          # Regularization parameter
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
                    'gamma': ['scale', 'auto'],                      # Kernel coefficient
                },
                "XGBClassifier": {
                    'learning_rate': [0.01, 0.1, 0.2],               # Step size
                    'n_estimators': [100, 200, 500],                 # Boosting rounds
                    'max_depth': [3, 6, 10],                         # Tree depth
                    'subsample': [0.7, 0.8, 0.9, 1.0],               # Sampling fraction
                    'colsample_bytree': [0.7, 0.8, 1.0],             # Feature fraction
                    'min_child_weight': [1, 3, 5],                   # Minimum child weight
                },
                "CatBoostClassifier": {
                    'depth': [6, 8, 10],                             # Tree depth
                    'learning_rate': [0.01, 0.1, 0.2],               # Step size
                    'iterations': [100, 200, 500],                   # Boosting rounds
                    'l2_leaf_reg': [1, 3, 5],                        # Regularization term
                    'bagging_temperature': [0.0, 0.5, 1.0],         # Bagging randomness
                },
                "AdaBoostClassifier": {
                    'n_estimators': [50, 100, 200, 500],             # Boosting stages
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],          # Step size
                    'algorithm': ['SAMME', 'SAMME.R'],               # Boosting algorithm
                }
            }


            logging.info("Evaluating models with hyperparameter tuning")
            best_model_score, best_model_name, model_report = evaluate_model(
                X_train, y_train, X_test, y_test, models, params
            )

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            # Check if the best model score is satisfactory
            if best_model_score < 0.6:
                logging.warning(f"Best model score ({best_model_score}) is below threshold (0.6)")
                raise CustomException("No Best Model Found")
            
            # Save the best model
            best_model = models[best_model_name]
            save_obj(self.model_trainer_config.model_file_path, best_model)

            logging.info(f"Model saved successfully at {self.model_trainer_config.model_file_path}")
        
    
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)

        return best_model_score
