from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Connection parameters for a single PostgreSQL database.
    All values are read from environment variables so nothing sensitive
    ever lives in source code.

    Use ``from_env()`` with a prefix to configure two separate databases
    from the same environment:

    Source database (the one being introspected):
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SCHEMA

    Repository database (where the KG is persisted):
        REPO_DB_HOST, REPO_DB_PORT, REPO_DB_NAME,
        REPO_DB_USER, REPO_DB_PASSWORD, REPO_DB_SCHEMA
    """

    host: str
    port: int
    dbname: str
    user: str
    password: str
    schema_name: str = "public"

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls, prefix: str = "DB") -> "DatabaseConfig":
        """
        Build a DatabaseConfig from environment variables sharing *prefix*.

        Variable names are formed as ``{PREFIX}_{FIELD}``, e.g.:
          - prefix="DB"      → DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SCHEMA
          - prefix="REPO_DB" → REPO_DB_HOST, REPO_DB_PORT, …

        Parameters
        ----------
        prefix:
            Environment-variable prefix (default ``"DB"``).

        Raises
        ------
        EnvironmentError
            If any required variable is missing or empty, or if
            ``{prefix}_PORT`` cannot be parsed as an integer.
        """
        missing: list[str] = []

        def _require(key: str) -> str:
            val = os.getenv(key, "").strip()
            if not val:
                missing.append(key)
            return val

        host = os.getenv(f"{prefix}_HOST", "localhost").strip()
        port_str = os.getenv(f"{prefix}_PORT", "5432").strip()
        dbname = _require(f"{prefix}_NAME")
        user = _require(f"{prefix}_USER")
        password = _require(f"{prefix}_PASSWORD")
        schema_name = os.getenv(f"{prefix}_SCHEMA", "public").strip()

        if missing:
            raise EnvironmentError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            )

        try:
            port = int(port_str)
        except ValueError:
            raise EnvironmentError(
                f"{prefix}_PORT must be an integer, got: {port_str!r}"
            )

        return cls(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            schema_name=schema_name,
        )

    @classmethod
    def source_db_from_env(cls) -> "DatabaseConfig":
        """Shorthand: read source-database config from ``DB_*`` variables."""
        return cls.from_env(prefix="DB")

    @classmethod
    def repo_db_from_env(cls) -> "DatabaseConfig":
        """Shorthand: read repository-database config from ``REPO_DB_*`` variables."""
        return cls.from_env(prefix="REPO_DB")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def dsn(self) -> str:
        """Return a libpq-compatible DSN string (password included)."""
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


@dataclass(frozen=True)
class ChromaConfig:
    """
    Configuration for the Chroma persistent vector store.

    Environment variable
    --------------------
    CHROMA_PERSIST_DIR   Path to the directory where Chroma stores its data.
                         (default: ``./chroma_db``)
    """

    persist_directory: str = "./chroma_db"

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "ChromaConfig":
        """Build a ChromaConfig from environment variables."""
        return cls(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db").strip(),
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"ChromaConfig(persist_directory={self.persist_directory!r})"
