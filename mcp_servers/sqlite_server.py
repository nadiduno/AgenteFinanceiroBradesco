import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TABLES = frozenset({'clientes', 'transacoes'})
MAX_ROWS = 500
SELECT_QUERY = re.compile(
    r'\A\s*(?:(?:--[^\r\n]*(?:\r?\n|$))|(?:/\*.*?\*/))*\s*(SELECT|WITH)\b',
    re.IGNORECASE | re.DOTALL,
)

load_dotenv(PROJECT_ROOT / '.env')

mcp = FastMCP('SQLite Read-Only')


def _database_path() -> Path:
    database_url = os.getenv('DATABASE_URL', 'sqlite:///data/transacoes.db')
    prefix = 'sqlite:///'
    if not database_url.startswith(prefix):
        raise ValueError('DATABASE_URL deve apontar para um arquivo SQLite local.')

    path = Path(database_url[len(prefix):])
    return path if path.is_absolute() else PROJECT_ROOT / path


def _connect_readonly() -> sqlite3.Connection:
    path = _database_path()
    if not path.is_file():
        raise FileNotFoundError(f'Banco SQLite não encontrado: {path}')

    connection = sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only = ON')
    return connection


def _authorize_readonly(action, arg1, _arg2, _database_name, _trigger_name):
    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    if action in allowed_actions:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and arg1 in ALLOWED_TABLES:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


@mcp.tool
def describe_schema() -> list[dict[str, object]]:
    """Descreve as tabelas permitidas, sem expor os rótulos de fraude."""
    with closing(_connect_readonly()) as connection:
        schema = []
        for table_name in sorted(ALLOWED_TABLES):
            columns = connection.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()
            schema.append({
                'table': table_name,
                'columns': [
                    {
                        'name': column['name'],
                        'type': column['type'],
                        'not_null': bool(column['notnull']),
                        'primary_key': bool(column['pk']),
                    }
                    for column in columns
                ],
            })
    return schema


@mcp.tool
def run_select(query: str, limit: int = 100) -> dict[str, object]:
    """Executa uma consulta somente leitura nas tabelas permitidas."""
    if not SELECT_QUERY.match(query):
        raise ValueError('A consulta deve começar com SELECT ou WITH ... SELECT.')
    if not 1 <= limit <= MAX_ROWS:
        raise ValueError(f'limit deve estar entre 1 e {MAX_ROWS}.')

    with closing(_connect_readonly()) as connection:
        connection.set_authorizer(_authorize_readonly)
        cursor = connection.execute(query)
        if cursor.description is None:
            raise ValueError('A consulta não retornou um conjunto de resultados.')

        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchmany(limit + 1)

    truncated = len(rows) > limit
    return {
        'columns': columns,
        'rows': [dict(row) for row in rows[:limit]],
        'truncated': truncated,
    }


if __name__ == '__main__':
    mcp.run()
