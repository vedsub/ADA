from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Connection parameters for a single PostgreSQL source database.
    All values are read from environment variables so nothing sensitive
    ever lives in source code.

    Environment variables
    ---------------------
    DB_HOST         Hostname / IP of the Postgres server   (default: localhost)
    DB_PORT         Port number                             (default: 5432)
    DB_NAME         Database name                          (required)
    DB_USER         Login role                             (required)
    DB_PASSWORD     Password                               (required)
    DB_SCHEMA       Schema to introspect                   (default: public)
    """

    host: str
    port: int
    dbname: str
    user: str
    password: str
    schema_name: str = "public"

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """
        Build a DatabaseConfig from environment variables.

        Raises
        ------
        EnvironmentError
            If any required variable is missing or empty.
        """
        missing: list[str] = []

        def _require(key: str) -> str:
            val = os.getenv(key, "").strip()
            if not val:
                missing.append(key)
            return val

        host = os.getenv("DB_HOST", "localhost").strip()
        port_str = os.getenv("DB_PORT", "5432").strip()
        dbname = _require("DB_NAME")
        user = _require("DB_USER")
        password = _require("DB_PASSWORD")
        schema_name = os.getenv("DB_SCHEMA", "public").strip()

        if missing:
            raise EnvironmentError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            )

        try:
            port = int(port_str)
        except ValueError:
            raise EnvironmentError(f"DB_PORT must be an integer, got: {port_str!r}")

        return cls(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            schema_name=schema_name,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def dsn(self) -> str:
        """Return a libpq-compatible DSN string (password is included)."""
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )

    def safe_repr(self) -> str:
        """Return a loggable representation that omits the password."""
        return (
            f"DatabaseConfig(host={self.host!r}, port={self.port}, "
            f"dbname={self.dbname!r}, user={self.user!r}, "
            f"schema_name={self.schema_name!r})"
        )

    def __repr__(self) -> str:  # never leak the password in repr
        return self.safe_repr()
