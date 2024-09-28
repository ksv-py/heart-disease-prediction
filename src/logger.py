import logging
import os
from datetime import datetime

# Define the log file name using the current date and time for uniqueness.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # Example: "09_20_2024_14_30_00.log"

# Set the path for the logs directory within the current working directory.
logs_path = os.path.join(os.getcwd(), "logs")  # Creates a "logs" folder in the current directory.

# Create the logs directory if it does not exist.
os.makedirs(logs_path, exist_ok=True)  # `exist_ok=True` ensures no error if the folder already exists.

# Full path to the log file where logs will be stored.
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging settings.
logging.basicConfig(
    filename=LOG_FILE_PATH,  # File to which logs will be written.
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Format of the log messages.
    level=logging.INFO  # Set the logging level to INFO, capturing all messages at this level and above.
)