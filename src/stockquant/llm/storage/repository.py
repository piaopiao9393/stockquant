"""因子仓库 - 持久化存储 LLM 挖掘的因子"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from stockquant.llm.core import FactorAST


@dataclass
class StoredFactor:
    """存储的因子记录"""

    id: int
    name: str
    ast_json: str
    description: str
    hypothesis: str
    direction: int
    metrics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    performance_summary: Dict[str, Any]
    is_active: bool = True

    def to_factor_ast(self) -> Optional[FactorAST]:
        """转换为 FactorAST 对象"""
        try:
            from stockquant.llm.core import ASTNode, NodeType

            ast_data = json.loads(self.ast_json)

            def parse_node(data: Dict) -> ASTNode:
                node_type = NodeType[data.get("type", "DATA")]
                params = data.get("params", {})
                children = [parse_node(c) for c in data.get("children", [])]
                return ASTNode(node_type=node_type, children=children, params=params)

            ast = parse_node(ast_data)

            factor = FactorAST(
                name=self.name,
                ast=ast,
                description=self.description,
                hypothesis=self.hypothesis,
                direction=self.direction,
            )
            factor.metrics = self.metrics
            return factor
        except Exception as e:
            logger.error(f"解析存储的因子失败: {e}")
            return None


class FactorRepository:
    """因子仓库 - SQLite 存储"""

    def __init__(self, db_path: str = "data/llm_factors.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    ast_json TEXT NOT NULL,
                    description TEXT,
                    hypothesis TEXT,
                    direction INTEGER DEFAULT 1,
                    metrics TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    performance_summary TEXT DEFAULT '{}',
                    is_active INTEGER DEFAULT 1
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    ic REAL,
                    return REAL,
                    rank_ic REAL,
                    FOREIGN KEY (factor_id) REFERENCES factors(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_factor_name ON factors(name)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_performance_date ON factor_performance(date)
            """)

            conn.commit()

    def save_factor(
        self,
        factor: FactorAST,
        performance_summary: Optional[Dict] = None,
    ) -> int:
        """保存因子

        Returns:
            因子 ID
        """
        ast_json = json.dumps(factor.ast.to_dict())
        metrics_json = json.dumps(factor.metrics)
        perf_json = json.dumps(performance_summary or {})

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO factors
                (name, ast_json, description, hypothesis, direction, metrics, performance_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    factor.name,
                    ast_json,
                    factor.description,
                    factor.hypothesis,
                    factor.direction,
                    metrics_json,
                    perf_json,
                ),
            )
            conn.commit()

            # 获取 ID
            cursor = conn.execute(
                "SELECT id FROM factors WHERE name = ?", (factor.name,)
            )
            factor_id = cursor.fetchone()[0]

        logger.info(f"保存因子: {factor.name} (ID: {factor_id})")
        return factor_id

    def get_factor(self, name: str) -> Optional[StoredFactor]:
        """获取单个因子"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM factors WHERE name = ? AND is_active = 1",
                (name,),
            ).fetchone()

            if row:
                return self._row_to_stored_factor(row)
        return None

    def get_factor_by_id(self, factor_id: int) -> Optional[StoredFactor]:
        """通过 ID 获取因子"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM factors WHERE id = ? AND is_active = 1",
                (factor_id,),
            ).fetchone()

            if row:
                return self._row_to_stored_factor(row)
        return None

    def list_factors(
        self,
        active_only: bool = True,
        min_ic: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[StoredFactor]:
        """列出所有因子"""
        query = "SELECT * FROM factors WHERE 1=1"
        params = []

        if active_only:
            query += " AND is_active = 1"

        query += " ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

            factors = [self._row_to_stored_factor(row) for row in rows]

        # 内存过滤 IC
        if min_ic is not None:
            factors = [
                f for f in factors
                if f.metrics.get("ic_mean", 0) >= min_ic
            ]

        return factors

    def deactivate_factor(self, name: str) -> bool:
        """停用因子"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE factors SET is_active = 0 WHERE name = ?",
                (name,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def activate_factor(self, name: str) -> bool:
        """激活因子"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE factors SET is_active = 1 WHERE name = ?",
                (name,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_factor(self, name: str) -> bool:
        """删除因子"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM factors WHERE name = ?",
                (name,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def record_performance(
        self,
        factor_id: int,
        date: str,
        ic: Optional[float] = None,
        ret: Optional[float] = None,
        rank_ic: Optional[float] = None,
    ) -> None:
        """记录因子每日表现"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO factor_performance (factor_id, date, ic, return, rank_ic)
                VALUES (?, ?, ?, ?, ?)
                """,
                (factor_id, date, ic, ret, rank_ic),
            )
            conn.commit()

    def get_performance_history(
        self,
        factor_id: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取因子历史表现"""
        query = "SELECT * FROM factor_performance WHERE factor_id = ?"
        params = [factor_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_similar_factors(
        self,
        factor: FactorAST,
        threshold: float = 0.85,
    ) -> List[Tuple[StoredFactor, float]]:
        """获取与给定因子相似的存储因子"""
        from stockquant.llm.core import calculate_similarity

        candidates = self.list_factors(active_only=True)
        similar = []

        for stored in candidates:
            stored_ast = stored.to_factor_ast()
            if stored_ast:
                sim = calculate_similarity(factor.ast, stored_ast.ast)
                if sim >= threshold:
                    similar.append((stored, sim))

        return sorted(similar, key=lambda x: x[1], reverse=True)

    def get_factor_statistics(self) -> Dict[str, Any]:
        """获取因子仓库统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            # 总数
            total = conn.execute(
                "SELECT COUNT(*) FROM factors"
            ).fetchone()[0]

            active = conn.execute(
                "SELECT COUNT(*) FROM factors WHERE is_active = 1"
            ).fetchone()[0]

            # 平均 IC
            avg_ic = conn.execute(
                """
                SELECT AVG(
                    json_extract(metrics, '$.ic_mean')
                ) FROM factors WHERE is_active = 1
                """
            ).fetchone()[0]

            # 最佳因子
            best = conn.execute(
                """
                SELECT name, json_extract(metrics, '$.ic_mean') as ic
                FROM factors WHERE is_active = 1
                ORDER BY ic DESC LIMIT 1
                """
            ).fetchone()

        return {
            "total_factors": total,
            "active_factors": active,
            "inactive_factors": total - active,
            "average_ic": avg_ic or 0,
            "best_factor": {"name": best[0], "ic": best[1]} if best else None,
        }

    def export_factors(self, output_path: str, active_only: bool = True) -> None:
        """导出因子到 JSON 文件"""
        factors = self.list_factors(active_only=active_only)

        export_data = {
            "export_time": datetime.now().isoformat(),
            "count": len(factors),
            "factors": [
                {
                    "name": f.name,
                    "description": f.description,
                    "hypothesis": f.hypothesis,
                    "direction": f.direction,
                    "metrics": f.metrics,
                    "ast": json.loads(f.ast_json),
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in factors
            ],
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"导出 {len(factors)} 个因子到 {output_path}")

    def import_factors(self, input_path: str, activate: bool = True) -> int:
        """从 JSON 文件导入因子"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for factor_data in data.get("factors", []):
            try:
                from stockquant.llm.core import ASTNode, NodeType

                ast_data = factor_data["ast"]

                def parse_node(data: Dict) -> ASTNode:
                    node_type = NodeType[data.get("type", "DATA")]
                    params = data.get("params", {})
                    children = [parse_node(c) for c in data.get("children", [])]
                    return ASTNode(node_type=node_type, children=children, params=params)

                ast = parse_node(ast_data)

                factor = FactorAST(
                    name=factor_data["name"],
                    ast=ast,
                    description=factor_data.get("description", ""),
                    hypothesis=factor_data.get("hypothesis", ""),
                    direction=factor_data.get("direction", 1),
                )
                factor.metrics = factor_data.get("metrics", {})

                self.save_factor(factor)
                count += 1

            except Exception as e:
                logger.warning(f"导入因子失败: {e}")

        logger.info(f"成功导入 {count} 个因子")
        return count

    def _row_to_stored_factor(self, row: sqlite3.Row) -> StoredFactor:
        """将数据库行转换为 StoredFactor"""
        return StoredFactor(
            id=row["id"],
            name=row["name"],
            ast_json=row["ast_json"],
            description=row["description"] or "",
            hypothesis=row["hypothesis"] or "",
            direction=row["direction"],
            metrics=json.loads(row["metrics"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"])
            if row["updated_at"] else None,
            performance_summary=json.loads(row["performance_summary"] or "{}"),
            is_active=bool(row["is_active"]),
        )
