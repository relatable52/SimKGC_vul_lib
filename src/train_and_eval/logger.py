import logging

LOGGER_NAME = "vul_lib_logger"

def _setup_logger():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for log in loggers:
        if "transformers" in log.name.lower():
            log.setLevel(logging.ERROR)

    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger

logger = _setup_logger()