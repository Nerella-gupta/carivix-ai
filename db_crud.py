from __future__ import annotations

import sqlite3
from typing import Any, Dict, Iterable, List, Optional


class DatabaseManager:
    """Small sqlite-backed CRUD helper used by the project tests."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        if not table_name:
            raise ValueError("table_name is required")
        columns_sql = ", ".join(f"{name} {dtype}" for name, dtype in schema.items())
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql})")
        self.conn.commit()

    def insert(self, table_name: str, row: Dict[str, Any]) -> int:
        columns = list(row.keys())
        placeholders = ", ".join("?" for _ in columns)
        values = [row[col] for col in columns]
        cursor = self.conn.execute(
            f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def insert_many(self, table_name: str, rows: Iterable[Dict[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        first = rows[0]
        columns = list(first.keys())
        placeholders = ", ".join("?" for _ in columns)
        values = [tuple(row[col] for col in columns) for row in rows]
        self.conn.executemany(
            f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        return len(rows)

    def read(self, table_name: str, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        sql = f"SELECT * FROM {table_name}"
        params: List[Any] = []
        if where:
            clauses = [f"{key} = ?" for key in where]
            sql += " WHERE " + " AND ".join(clauses)
            params.extend(where[key] for key in where)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def read_one(self, table_name: str, where: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        rows = self.read(table_name, where=where)
        return rows[0] if rows else None

    def update(self, table_name: str, values: Dict[str, Any], where: Optional[Dict[str, Any]] = None) -> int:
        if not values:
            return 0
        assignments = ", ".join(f"{key} = ?" for key in values)
        params = list(values.values())
        sql = f"UPDATE {table_name} SET {assignments}"
        if where:
            where_sql = " AND ".join(f"{key} = ?" for key in where)
            sql += " WHERE " + where_sql
            params.extend(where[key] for key in where)
        cursor = self.conn.execute(sql, params)
        self.conn.commit()
        return int(cursor.rowcount)

    def delete(self, table_name: str, where: Optional[Dict[str, Any]] = None) -> int:
        sql = f"DELETE FROM {table_name}"
        params: List[Any] = []
        if where:
            where_sql = " AND ".join(f"{key} = ?" for key in where)
            sql += " WHERE " + where_sql
            params.extend(where[key] for key in where)
        cursor = self.conn.execute(sql, params)
        self.conn.commit()
        return int(cursor.rowcount)

    def count(self, table_name: str, where: Optional[Dict[str, Any]] = None) -> int:
        sql = f"SELECT COUNT(*) AS count FROM {table_name}"
        params: List[Any] = []
        if where:
            where_sql = " AND ".join(f"{key} = ?" for key in where)
            sql += " WHERE " + where_sql
            params.extend(where[key] for key in where)
        row = self.conn.execute(sql, params).fetchone()
        return int(row["count"]) if row else 0

    def close(self) -> None:
        self.conn.close()
