import logging
import os
from datetime import datetime  

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

artifacts_dir = os.path.join(os.getcwd(), "artifacts")
logs_dir=os.path.join(artifacts_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)  

log_file_path=os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()