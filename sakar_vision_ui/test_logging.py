
from utils.logging import setup_logging, get_logger, shutdown_logging

# Initialize logging system
setup_logging()
logger = get_logger("test_logger")

# Log some test messages
logger.info("This is a test info message")
logger.warning("This is a test warning message")
logger.error("This is a test error message")

print("Test complete! Check for logs directory")
shutdown_logging()

