import logging
import threading

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_LEVEL = logging.INFO


def configure_logging(level: int = DEFAULT_LOG_LEVEL) -> None:
    """Configure the root logger with a shared format and level."""
    logging.basicConfig(level=level, format=LOG_FORMAT)


def log_session(msg: str, session_obj: object, logger_name: str) -> None:
    """
    Log the current thread id plus the session object's id.

    Args:
        msg: Description of the step being traced.
        session_obj: The SQLModel/SQLAlchemy session instance.
        logger_name: Name of the logger to emit through (pass __name__ from caller).
    """
    logger = logging.getLogger(logger_name)
    logger.info(f"[tid={threading.get_ident()}] {msg} session_obj_id={id(session_obj)}")

