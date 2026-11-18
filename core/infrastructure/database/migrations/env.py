from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from core.infrastructure.database.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline():
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    # Use an explicit SQLAlchemy transaction. SQLite reports transactional_ddl=False
    # to Alembic by default; without this outer transaction the DDL may be applied
    # while alembic_version's INSERT is rolled back when the connection closes.
    # That leaves a fully-created schema with no revision marker and causes the next
    # startup to replay 0001_initial_schema (table cameras already exists).
    with connectable.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            transactional_ddl=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
