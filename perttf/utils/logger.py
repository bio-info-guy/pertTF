import logging
import sys
import logging
import sys

def create_logger(id=None, log_file=None):
    logger_name = f"pertTF_worker_{id}" if id is not None else 'pertTF'
    logger = logging.getLogger(logger_name)
    # check if logger has been initialized
    if not logger.hasHandlers() or len(logger.handlers) == 0:
        logger.propagate = False
        logger.setLevel(logging.INFO)
        # Route to file if specified, otherwise default to stdout
        if log_file:
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger