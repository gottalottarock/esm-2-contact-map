import logging
import coloredlogs

def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(level=log_level)
    return logging.getLogger(__name__)