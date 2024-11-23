import os
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
sys.path.append(str(Path(__file__).parent.parent))
from utils import save_obj
from logger import logging
from exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd


@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    Specifies the path where the preprocessor object will be saved.
    """
    processor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    """
    A class responsible for handling data transformation, including:
    - Configuring pipelines for preprocessing
    - Splitting data into features and target variables
    - Saving the preprocessor object for later use
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a preprocessor object with pipelines for:
        - Imputation using most frequent strategy
        - OneHotEncoding for categorical variables
        - Scaling numerical variables with mean disabled

        Returns:
        --------
        preprocessor: sklearn.compose.ColumnTransformer
            A column transformer object configured with preprocessing steps.
        """
        try:
            logging.info("Creating preprocessing pipeline")

            # Define a pipeline with imputation, encoding, and scaling
            pipeline = Pipeline(
                steps=[
                    ('Imputer', SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder', OneHotEncoder(sparse_output=True)),
                    # ('Standard Scaler', StandardScaler(with_mean=False))
                ]
            )

            # Define the list of features to transform
            features = [
                'HighBP', 'HighChol', 'BMI', 'Smoker',
                'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies',
                'HvyAlcoholConsump', 'AnyHealthcare', 'GenHlth', 'DiffWalk', 'Sex',
                'Mental_Health_Category', 'Physical_Health_Category'
            ]

            # Create a ColumnTransformer with the defined pipeline
            preprocessor = ColumnTransformer([('pipeline', pipeline, features)])
            logging.info("Preprocessing pipeline created successfully")

            return preprocessor

        except Exception as e:
            logging.error("Error occurred while creating the preprocessing pipeline")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Handles the transformation of data by:
        - Reading train and test datasets
        - Splitting features and target variables
        - Applying preprocessing transformations
        - Saving the preprocessor object for later use

        Parameters:
        -----------
        train_path: str
            Path to the training data CSV file.
        test_path: str
            Path to the testing data CSV file.

        Returns:
        --------
        tuple: (np.ndarray, np.ndarray)
            Transformed train and test datasets as numpy arrays.
        """
        try:
            logging.info("Initiating data transformation process")

            # Read the training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Training data loaded successfully from {train_path}")
            logging.info(f"Testing data loaded successfully from {test_path}")

            # Log shape and column details of datasets
            logging.info(f"Training data shape: {train_df.shape}, Columns: {train_df.columns.tolist()}")
            logging.info(f"Testing data shape: {test_df.shape}, Columns: {test_df.columns.tolist()}")

            # Obtain preprocessing object
            logging.info("Obtaining preprocessing pipeline")
            preprocessing_obj = self.get_data_transformer_object()
            target_column = 'HeartDiseaseorAttack'  # Define the target column

            # Splitting features and target variable for training data
            logging.info("Splitting features and target variable for training data")
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            # Splitting features and target variable for testing data
            logging.info("Splitting features and target variable for testing data")
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            # Log dataset shapes after splitting
            logging.info(f"Train features shape: {input_feature_train_df.shape}, Train target shape: {target_feature_train_df.shape}")
            logging.info(f"Test features shape: {input_feature_test_df.shape}, Test target shape: {target_feature_test_df.shape}")

            # Apply transformations to training and testing datasets
            logging.info("Applying preprocessing transformations on training data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info("Transformations on training data completed")

            logging.info("Applying preprocessing transformations on testing data")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Transformations on testing data completed")

            # Combine transformed input features with target variables
            logging.info("Combining transformed features with target variables")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object
            logging.info("Saving the preprocessing object")
            save_obj(
                file_path=self.data_transformation_config.processor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Preprocessing object saved at {self.data_transformation_config.processor_obj_file_path}")

            return train_arr, test_arr

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys)
