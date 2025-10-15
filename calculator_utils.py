# -*- coding: utf-8 -*-
"""
calculator_utils.py

提供 Calculator Agent 的底层函数：
- ensure_df: 从 state 中加载 CSV
- apply_filters: 根据条件筛选 DataFrame
- most_similar_column: 模糊列名解析
- md_preview: Markdown 格式表格预览
"""

import io
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
from difflib import get_close_matches


def ensure_df(state: Dict[str, Any]) -> pd.DataFrame:
    """从 QueryState 中加载 CSV 为 DataFrame。支持 csv_df / csv_bytes / csv_text / csv_path / uploaded_file。"""
    df = state.get("csv_df")
    if isinstance(df, pd.DataFrame):
        return df

    if state.get("csv_bytes"):
        return pd.read_csv(io.BytesIO(state["csv_bytes"]))

    if state.get("csv_text"):
        return pd.read_csv(io.StringIO(state["csv_text"]))

    if state.get("csv_path"):
        return pd.read_csv(state["csv_path"])

    uf = state.get("uploaded_file")
    if uf is not None:
        if isinstance(uf, dict) and "bytes" in uf:
            return pd.read_csv(io.BytesIO(uf["bytes"]))
        if hasattr(uf, "read"):
            return pd.read_csv(io.BytesIO(uf.read()))

    raise ValueError("No CSV found on state (expected csv_df/csv_bytes/csv_text/csv_path/uploaded_file).")


def apply_filters(df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
    if not filters:
        return df
    out = df.copy()
    mask_all = pd.Series(True, index=out.index)

    for f in filters:
        col = f.get("column")
        op  = (f.get("op") or "").strip().lower()
        val = f.get("value")
        if col not in out.columns:
            raise ValueError(f"Column not found: {col}")

        series = out[col]
        left  = pd.to_numeric(series, errors="coerce")
        if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
            right = pd.to_numeric(pd.Series(val, index=out.index), errors="coerce")
        else:
            right = pd.to_numeric(pd.Series([val]*len(out), index=out.index), errors="coerce")

        can_numeric = left.notna().any() and right.notna().any()
        if can_numeric and op in {"<","<="," >",">=","==","!="}:
            mask = {
                "<":  left <  right,
                "<=": left <= right,
                ">":  left >  right,
                ">=": left >= right,
                "==": left == right,
                "!=": left != right,
            }[op]
        else:
            s = series.astype(str).str.strip().str.lower()
            if isinstance(val, (list, tuple, set)):
                vals = [str(v).strip().lower() for v in val]
                if   op in {"in","isin","=="}:   mask = s.isin(vals)
                elif op in {"not in","notin","!="}: mask = ~s.isin(vals)
                else: raise ValueError("string op must be in/isin/==/!= for list-like values")
            else:
                rv = str(val).strip().lower()
                if   op == "contains":          # 如需“包含”，显式用 contains
                    mask = s.str.contains(re.escape(rv), na=False)
                elif op == "==":
                    mask = (s == rv)
                elif op == "!=":
                    mask = (s != rv)
                elif op in {"<","<=",">",">="}: # 字典序比较（少用）
                    mask = eval(f"s {op} rv")
                else:
                    raise ValueError("string op must be one of contains/==/!=/<,<=,>,>=")

        mask_all &= mask.fillna(False)

    return out.loc[mask_all]



def normalize_col(s: str) -> str:
    """标准化列名：去除下划线、连字符、小写化。"""
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def most_similar_column(user_col: str, columns: List[str], min_ratio: float = 0.6) -> Tuple[Optional[str], List[str]]:
    """根据模糊列名返回最相似的真实列名 (best, candidates)。"""
    if not user_col:
        return None, []
    norm_user = normalize_col(user_col)
    norm_map = {c: normalize_col(c) for c in columns}
    for real, norm in norm_map.items():
        if norm == norm_user:
            return real, []
    cands_norm = get_close_matches(norm_user, list(norm_map.values()), n=5, cutoff=min_ratio)
    rev = {v: k for k, v in norm_map.items()}
    cands = [rev[nc] for nc in cands_norm if nc in rev]
    best = cands[0] if cands else None
    return best, cands


def md_preview(df: pd.DataFrame, n: int = 5) -> str:
    """生成 DataFrame 前 n 行的 Markdown 表格文本。"""
    head = df.head(n)
    return f"Rows={len(df)}, Cols={len(df.columns)}\n\n" + head.to_markdown(index=False)
