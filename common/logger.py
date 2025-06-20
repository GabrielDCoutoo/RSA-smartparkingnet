import logging
import json
from datetime import datetime


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "lvl": record.levelname,
            "time": datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
            "msg": record.getMessage()
        }
        return json.dumps(log_record)


# Custom level definition function
def add_logging_level(level_name, level_num, method_name=None):
    if not method_name:
        method_name = level_name.lower()

    if hasattr(logging, level_name):
        raise AttributeError(f'{level_name} already defined in logging module')
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError(f'{method_name} already defined in logger class')

    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging.getLoggerClass(), method_name, log_for_level)
    setattr(logging, method_name, log_to_root)

# Add the TMSTMP level with a numeric value of 5
add_logging_level('TMSTMP', 5)


def setup_logger(name, level):
    """Configure and return a logger with the specified name, ensure no duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # Check if handlers are already added
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
    logger.propagate = False  # Prevent the logger from propagating messages to the parent logger
    return logger


def getLevelName(level):
    return logging.getLevelName(level)
