import sys
from loguru import logger

logger = logger
logger.configure(
    handlers=[
        {
            "sink": sys.stderr,
            "format": "<d>{extra}</> | "
            + "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | "
            + "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            + "<level>{message}</level>",
            "serialize": False,
        }
    ]
)

handle_catch_error = logger.catch(onerror=lambda _: sys.exit(-1))
