import sys

# Function to get detailed error message with script name, line number, and error message.
def error_message_detail(error, error_detail: sys):
    """
    Extracts and formats detailed error information including the filename, line number, 
    and error message where the exception occurred.

    Args:
        error (Exception): The exception object containing the error message.
        error_detail (sys): The sys module, used to access traceback details of the exception.

    Returns:
        str: A formatted string containing the error message with details such as the 
             filename and line number where the error occurred.
    
    Example:
        try:
            # Code that raises an exception
            result = 10 / 0
        except Exception as e:
            # Capture detailed error message
            detailed_error = error_message_detail(e, sys)
            print(detailed_error)
            # Output: Error occurred in Python script name [your_script.py] at line no. [X] -- error message: [division by zero]
    """
    _, _, exc_tb = error_detail.exc_info()
    # Get the filename where the error occurred.
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Format the error message with filename, line number, and the actual error message.
    error_message = "Error occurred in Python script name [{0}] at line no. [{1}] -- error message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

# Custom exception class for detailed error handling.
class CustomException(Exception):
    # Constructor that takes the error message and error details (sys module).
    def __init__(self, error_message, error_detail: sys):
        # Initialize the base Exception class with the error message.
        super().__init__(error_message)
        # Generate a detailed error message using the provided function.
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    # Override the string representation of the exception to return the detailed error message.
    def __str__(self):
        return self.error_message