import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))
from components.model_trainer import ModelTrainer
from exception import CustomException  # Custom exception handling class
from logger import logging  # Logging setup
from data_transformation import DataTransformation

# Define a dataclass for configuring the data ingestion paths
@dataclass
class DataIngestionConfig:
    # Paths to save raw, train, and test data
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    """
    This class is responsible for ingesting the data, splitting it into
    training and test sets, and saving these sets to defined paths.
    """
    def __init__(self):
        # Initialize the data ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This function handles the process of data ingestion which includes:
        - Reading the raw dataset
        - Splitting it into training and testing datasets
        - Saving these datasets as CSV files
        
        Returns:
        --------
        tuple: Paths of the train and test datasets.
        """
        logging.info("Starting data ingestion process.")
        try:
            # Reading the raw dataset
            
            df = pd.read_csv('notebook/data/eda_heart_data.csv')
            logging.info("Successfully read the dataset.")

            

            # Creating the directory for saving raw data if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            # Splitting the dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Split the data into training and testing sets.")

            # Creating the directories and saving the train and test datasets
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f"Training data saved at: {self.ingestion_config.train_data_path}")

            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Testing data saved at: {self.ingestion_config.test_data_path}")

            print(df.columns)
            print(pd.read_csv(self.ingestion_config.train_data_path).columns)
            print(pd.read_csv(self.ingestion_config.test_data_path).columns)
            # Returning paths of train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise CustomException(e, sys)  # Raising custom exception with the system info

# Main execution block
if __name__ == "__main__":
    # Create an instance of DataIngestion and initiate data ingestion.
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr = data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(f"Model Accuracy: {modeltrainer.initiate_model_training(train_arr,test_arr)}")