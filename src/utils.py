import os
import sys
import dill
from pathlib import Path

from sklearn.model_selection import GridSearchCV, cross_val_score
sys.path.append(str(Path(__file__).parent.parent))
from logger import logging
from exception import CustomException

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
from sklearn.model_selection import GridSearchCV, cross_val_score

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple models using GridSearchCV for hyperparameter tuning and return the model with the best test accuracy.
    
    Parameters:
    - X_train: Features for training
    - y_train: Target variable for training
    - X_test: Features for testing
    - y_test: Target variable for testing
    - models: Dictionary of model names and instances
    - params: Dictionary of hyperparameters for each model
    
    Returns:
    - report: Dictionary with model names and their test accuracies
    """
    
    # Initialize the report dictionary and variables to track the best model and accuracy
    report = {}
    best_accuracy = 0
    best_model = None
    # Loop through models to evaluate each one
    for model_name, model_instance in models.items():
        try:
            # Log the start of evaluation for the model
            logging.info(f"Evaluating {model_name}...")

            # Retrieve hyperparameters for the current model
            model_params = params.get(model_name, {})
            
            # Apply GridSearchCV for hyperparameter tuning
            logging.info(f"Performing GridSearchCV for {model_name}...")
            gs = GridSearchCV(model_instance, model_params, cv=3, scoring='accuracy', n_jobs=7)
            gs.fit(X_train, y_train)

            # Log the best parameters found by GridSearchCV
            logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            print(gs.best_params_)
            # Set the best parameters found by GridSearchCV
            model_instance.set_params(**gs.best_params_)
            model_instance.fit(X_train, y_train)

            # Get cross-validation scores for both training and test sets
            train_model_accuracy = cross_val_score(model_instance, X_train, y_train, cv=5, scoring='accuracy').mean()
            test_model_accuracy = cross_val_score(model_instance, X_test, y_test, cv=5, scoring='accuracy').mean()

            # Log the accuracy for train and test sets
            logging.info(f"{model_name} - Train Accuracy: {train_model_accuracy:.4f}")
            logging.info(f"{model_name} - Test Accuracy: {test_model_accuracy:.4f}")

            # Store the test accuracy in the report
            report[model_name] = test_model_accuracy

            # Track the best model based on the highest test accuracy
            if test_model_accuracy > best_accuracy:
                best_accuracy = test_model_accuracy
                best_model = model_instance
                logging.info(f"{model_name} is currently the best model with a test accuracy of {best_accuracy:.4f}")

        except Exception as e:
            # Handle any exceptions that may occur during evaluation
            logging.error(f"Error evaluating {model_name}: {e}")
            raise e

    # Log the final report of all models' test accuracies
    logging.info("Model evaluation completed.")
    logging.info(f"Best Model: {best_model} with Accuracy of {best_accuracy}")
    logging.info(f"Final Report: {report}")

    # Return the report with model accuracies
    return best_accuracy, best_model, report
