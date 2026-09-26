import psycopg
from psycopg.rows import dict_row
from contextlib import contextmanager

from app.config import PG_DSN


@contextmanager
def get_conn():
    """Yield a Postgres connection with dict-row results.

    Usage:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM passages LIMIT 1")
                row = cur.fetchone()
    """
    conn = psycopg.connect(PG_DSN, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
