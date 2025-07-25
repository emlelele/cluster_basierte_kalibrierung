from __future__ import annotations

import sys

from contextlib import contextmanager

import saio

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

from mwe_db_access.config import settings


def register_schemas(engine: Engine) -> None:
    """
    Register database schemas using saio based on configuration settings.

    This function registers all schemas defined in the settings with the given database
    engine.

    Parameters
    ----------
    engine : Engine
        The SQLAlchemy engine instance used to connect to the database.

    Raises
    ------
    Exception
        If an error occurs during schema registration.

    """
    try:
        for schema in settings["database"]["SCHEMAS"]:
            saio.register_schema(schema, engine)
            logger.info(f"Schema {schema} registered successfully.")
    except Exception as e:
        logger.error(f"Failed to register schema: {e}")
        raise


def get_saio_obj(schema: str, table: str) -> DeclarativeMeta:
    """
    Get a SQLAlchemy model object dynamically from the saio module.

    Parameters
    ----------
    schema : str
        The schema name where the table is defined.
    table : str
        The table name for which the model object is required.

    Returns
    -------
    DeclarativeMeta
        The SQLAlchemy model class for the specified table.

    Raises
    ------
    AttributeError
        If the module or table attribute cannot be found.

    """
    try:
        return sys.modules[f"saio.{schema}"].__getattr__(table)
    except AttributeError as e:
        logger.error(
            f"Failed to get SQLAlchemy object for schema {schema} and table {table}: "
            f"{e}"
        )
        raise


def engine() -> Engine:
    """
    Create and return a SQLAlchemy engine instance using configuration settings.

    Returns
    -------
    Engine
        The SQLAlchemy engine connected to the database using credentials from settings.

    """
    cred = settings["database"]["credentials"]

    engine = create_engine(
        f"postgresql+psycopg2://{cred['POSTGRES_USER']}:"
        f"{cred['POSTGRES_PASSWORD']}@{cred['HOST']}:"
        f"{cred['PORT']}/{cred['POSTGRES_DB']}",
        echo=False,
    )

    logger.info("SQLAlchemy engine created successfully.")

    return engine


@contextmanager
def session_scope(engine: Engine) -> Session:
    """
    Provide a transactional scope around a series of operations.

    Parameters
    ----------
    engine : Engine
        The SQLAlchemy engine instance used to create a session.

    Yields
    ------
    session
        A SQLAlchemy session object.

    """
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Session rollback due to an exception: {e}")
        raise
    finally:
        session.close()
        logger.info("Session closed.")
