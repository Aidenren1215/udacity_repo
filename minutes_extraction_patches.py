END_MARKER_RE = re.compile(
    r"with\s+no\s+other\s+matters,\s*the\s+meeting\s+ended\s+at",
    re.IGNORECASE
)


def find_end_marker_page(doc: fitz.Document, start_from: int) -> int:
    """
    Find the first page whose ANY line (after normalize)
    contains 'with no other matters, the meeting ended at'
    """

    for pno in range(start_from, doc.page_count):
        try:
            raw = doc.load_page(pno).get_text("text")
        except Exception:
            # 防止 corrupt object stream，直接跳过这一页
            continue

        for line in raw.splitlines():
            ln = _normalize_text(line)
            if ln and END_MARKER_RE.search(ln):
                return pno

    raise ValueError(
        "End marker page not found: "
        "no line containing 'with no other matters, the meeting ended at'."
    )
