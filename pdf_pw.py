import os
import re
import fitz  # PyMuPDF

def open_pdf_auto_password(file_path: str) -> fitz.Document:
    """
    Open a PDF with PyMuPDF.
    - If it requires a password: extract 'Mon YY' (e.g., 'Sep 24') from filename anywhere,
      build password 'alcommmyy' (e.g., 'alcosep24'), authenticate, and return doc.
    - If it does not require a password: return doc directly.

    Caller is responsible for doc.close().
    """
    doc = fitz.open(file_path)

    # No password needed
    if not doc.needs_pass:
        return doc

    # Need password -> extract from filename
    filename = os.path.basename(file_path).lower()
    m = re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\s_\-]*([0-9]{2})\b",
        filename
    )
    if not m:
        doc.close()
        raise ValueError(
            f"PDF requires password, but cannot find 'Mon YY' (e.g., 'Sep 24') in filename: {filename}"
        )

    password = f"alco{m.group(1)}{m.group(2)}"

    if not doc.authenticate(password):
        doc.close()
        raise RuntimeError(
            f"PDF password incorrect for {os.path.basename(file_path)} (tried: {password})"
        )

    return doc
