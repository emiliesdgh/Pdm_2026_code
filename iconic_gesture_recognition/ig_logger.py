# --- ig_logger.py ---
import logging

def setup_logger(log_filename="session_log.txt"):
    # Create a custom logger
    logger = logging.getLogger("GestureLogger")
    logger.setLevel(logging.INFO)

    # To prevent adding multiple handlers if the function is called twice
    if not logger.handlers:
        # 1. Create a file handler (saves to a file)
        file_handler = logging.FileHandler(log_filename, mode='a') # 'a' is append mode
        
        # 2. Create a stream handler (prints to the terminal)
        console_handler = logging.StreamHandler()

        # 3. Create a formatter and add it to the handlers
        # This adds the exact date and time to every log!
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 4. Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger