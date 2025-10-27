"""Database migration utilities using Alembic."""

import os
from pathlib import Path
from alembic import command
from alembic.config import Config

from tx.tinker.db_models import get_database_url
from tx.utils.log import logger


def run_migrations_on_startup():
    """Run Alembic migrations to upgrade the database to the latest version.
    
    This function is called on API startup to ensure the database schema
    is up-to-date before the application begins processing requests.
    """
    # Get the directory containing this file (tx/tinker)
    tinker_dir = Path(__file__).parent
    alembic_ini_path = tinker_dir / "alembic.ini"
    
    if not alembic_ini_path.exists():
        logger.warning(f"Alembic configuration not found at {alembic_ini_path}, skipping migrations")
        return
    
    # Create Alembic config
    alembic_cfg = Config(str(alembic_ini_path))
    
    # Set the script location to the alembic directory
    alembic_cfg.set_main_option("script_location", str(tinker_dir / "alembic"))
    
    # Get the database URL from environment or config
    db_url = get_database_url()
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    
    logger.info(f"Running Alembic migrations for database: {db_url}")
    
    try:
        # Run migrations to upgrade to the latest version
        command.upgrade(alembic_cfg, "head")
        logger.info("Successfully applied all pending migrations")
    except Exception as e:
        logger.error(f"Error running migrations: {e}")
        raise

