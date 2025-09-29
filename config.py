import logging
from colorama import Fore, Style, init
import warnings
from pathlib import Path
# from __future__ import print_function


warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)


init(autoreset=True)


BASE_DIR = Path(__file__).resolve().parent
original_dataset = BASE_DIR / "datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv"

open_dataset = BASE_DIR / "datasets/open_dataset.csv"
encoded_dataset = BASE_DIR / "datasets/encoded_dataset.csv"
MODEL_DIR = BASE_DIR / "models/"


class CustomFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)
        log_message = super().format(record)
        return f"{log_color}{log_message}{Style.RESET_ALL}"


def getLogger():
    # Set up logging
    logger = logging.getLogger("colored_logger")
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter("- %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger
