"""
database.py — 数据库存储模块

提供三个核心函数：save_record / get_all_records / get_statistics
默认使用 SQLite 实现，通过抽象存储层设计，方便后续无缝切换为 MySQL、PostgreSQL 等。
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional

# ========== 数据库配置 ==========
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "diagnose.db")
_TABLE_NAME = "diagnose_records"


# ============================================================
# 抽象存储层 — 方便后续切换数据库引擎
# ============================================================

class StorageBackend:
    """
    存储后端抽象基类。
    后续如需切换到 MySQL / PostgreSQL 等，只需继承此类并实现对应方法，
    再修改 _get_backend() 返回值即可，上层调用代码无需任何改动。
    """

    def save(self, record: dict) -> None:
        raise NotImplementedError

    def get_all(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_count(self) -> int:
        raise NotImplementedError


class SQLiteBackend(StorageBackend):
    """SQLite 存储后端实现"""

    def __init__(self, db_path: str, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        self._init_table()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 让查询结果可通过列名访问
        return conn

    def _init_table(self) -> None:
        """
        初始化数据表（若不存在则创建）。
        字段设计兼容 record 字典至少包含：
          field_name, location, predicted_class, confidence, is_reliable, weather_risk
        额外增加 id 主键和 created_at 时间戳便于追溯。
        """
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            field_name      TEXT    DEFAULT '',
            location        TEXT    DEFAULT '',
            predicted_class TEXT    DEFAULT '',
            confidence      REAL    DEFAULT 0.0,
            is_reliable     INTEGER DEFAULT 0,
            weather_risk    TEXT    DEFAULT '',
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        conn = self._get_conn()
        try:
            conn.execute(sql)
            conn.commit()
        finally:
            conn.close()

    def save(self, record: dict) -> None:
        """保存单条诊断记录"""
        sql = f"""
        INSERT INTO {self.table_name}
            (field_name, location, predicted_class, confidence, is_reliable, weather_risk)
        VALUES
            (:field_name, :location, :predicted_class, :confidence, :is_reliable, :weather_risk)
        """
        # 从 record 字典中提取各字段，缺失则给默认值
        params = {
            "field_name": record.get("field_name", ""),
            "location": record.get("location", ""),
            "predicted_class": record.get("predicted_class", ""),
            "confidence": record.get("confidence", 0.0),
            "is_reliable": 1 if record.get("is_reliable", False) else 0,
            "weather_risk": record.get("weather_risk", ""),
        }
        conn = self._get_conn()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def get_all(self) -> List[Dict[str, Any]]:
        """返回全部诊断记录，每条记录为字典"""
        sql = f"""
        SELECT id, field_name, location, predicted_class,
               confidence, is_reliable, weather_risk, created_at
        FROM {self.table_name}
        ORDER BY id ASC
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
        finally:
            conn.close()

        # 将 sqlite3.Row 转换为普通字典
        records = []
        for row in rows:
            rec = {
                "id": row["id"],
                "field_name": row["field_name"],
                "location": row["location"],
                "predicted_class": row["predicted_class"],
                "confidence": row["confidence"],
                "is_reliable": bool(row["is_reliable"]),
                "weather_risk": row["weather_risk"],
                "created_at": row["created_at"],
            }
            records.append(rec)
        return records

    def get_count(self) -> int:
        """返回总记录数"""
        sql = f"SELECT COUNT(*) AS cnt FROM {self.table_name}"
        conn = self._get_conn()
        try:
            cursor = conn.execute(sql)
            row = cursor.fetchone()
        finally:
            conn.close()
        return row["cnt"] if row else 0


# ============================================================
# 后端工厂 — 切换数据库只需修改此处
# ============================================================

_backend: Optional[StorageBackend] = None


def _get_backend() -> StorageBackend:
    """
    获取存储后端单例。
    【切换数据库】只需将此处改为返回其他 Backend 实例即可，
    例如: return MySQLBackend(host, port, user, password, database)
    """
    global _backend
    if _backend is None:
        _backend = SQLiteBackend(_DB_PATH, _TABLE_NAME)
    return _backend


# ============================================================
# 对外暴露的三个函数（函数签名不可改动）
# ============================================================

def save_record(record: dict) -> None:
    """
    保存单条诊断记录。

    Args:
        record: 记录字典，至少包含字段：
            field_name, location, predicted_class,
            confidence, is_reliable, weather_risk

    Returns:
        None
    """
    try:
        backend = _get_backend()
        backend.save(record)
    except Exception as e:
        # 数据库写入失败时静默处理，避免上层异常
        print(f"[database] 保存记录失败: {e}")


def get_all_records() -> list[dict]:
    """
    返回全部存储的记录列表。

    Returns:
        list[dict]: 每条记录是一个字典，包含所有字段
    """
    try:
        backend = _get_backend()
        return backend.get_all()
    except Exception as e:
        print(f"[database] 查询记录失败: {e}")
        return []


def get_statistics() -> dict:
    """
    返回统计信息。

    Returns:
        dict: {"total_diagnose": 整数}，整数为总诊断记录条数
    """
    try:
        backend = _get_backend()
        count = backend.get_count()
        return {"total_diagnose": count}
    except Exception as e:
        print(f"[database] 统计查询失败: {e}")
        return {"total_diagnose": 0}


# ========== 模块自测 ==========
if __name__ == "__main__":
    print("=== 测试 save_record ===")
    test_record = {
        "field_name": "小麦叶片",
        "location": "西安",
        "predicted_class": "锈病",
        "confidence": 0.92,
        "is_reliable": True,
        "weather_risk": "high"
    }
    save_record(test_record)
    print("已保存一条测试记录")

    print("\n=== 测试 get_all_records ===")
    all_records = get_all_records()
    for r in all_records:
        print(r)

    print("\n=== 测试 get_statistics ===")
    stats = get_statistics()
    print(stats)
