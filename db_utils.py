"""
Database engine configuration and helper utilities.
"""
from collections.abc import Generator

from sqlalchemy import event
from sqlmodel import SQLModel, Session, create_engine


__all__ = ("engine", "create_db_and_tables", "get_db_session")


# Database engine configuration
DATABASE_FILE = "database.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"


# Create SQLAlchemy engine with SQLite-specific configurations
# - echo=False: Not logging all SQL statements (logging takes care of it for debugging)
# - check_same_thread=False: Allow connections to be used across async thread switches
#   (Required for FastAPI's async nature; see Python sqlite3 docs)
# - timeout=30: Wait up to 30 seconds for database lock before raising error
#   (Default is 5 seconds; increased to handle concurrent requests better)
engine = create_engine(
    DATABASE_URL, 
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 30}
)


# Understand the flow and when @event.listens_for triggers:
# 1. App starts
# 2. create_engine() is called
# 3. Engine created, but no connections yet
# (now every time a NEW connection opens, 4.-11. runs)
# 4. First HTTP request comes in
# 5. get_db_session() is called
# 6. Session tries to get a connection from the engine
# 7. Engine opens a NEW database connection
# 8. @event.listens_for triggers! ← HERE
# 9. _sqlite_set_pragmas() is called automatically
# 10. WAL mode is set
# 11. Connection is now ready to use
@event.listens_for(engine, "connect")
def _sqlite_set_pragmas(dbapi_connection, _connection_record):
    """
    Set pragmas on connection open:
    - Enable WAL (write-ahead log) for better concurrency.
    Note: WAL is persistent on the DB file; this sets it on each connection defensively.
    """
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.close()
    except Exception:
        # best-effort; we don't want to crash app creation if the pragma fails
        pass


def create_db_and_tables() -> None:
    """
    Create all tables defined on SQLModel metadata if they don't exist yet.
    Idempotent: calling multiple times will not overwrite existing tables.
    """
    SQLModel.metadata.create_all(engine)


# Dependency and Service functions
def get_db_session() -> Generator[Session, None, None]:
    """
    Provide a database connection for the current request.

    This is a SQLAlchemy connection (not a logical user session). It enables
    queries/inserts/updates and is auto-closed by FastAPI after the response.
    """
    with Session(engine) as db_session:
        # yield -> FastAPI will close the session after the response
        yield db_session