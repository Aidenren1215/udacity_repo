import re
import pdfplumber
from typing import List, Optional, Callable, Iterable

def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> List[str]:
    """
    Split a given text using a separator with the option to keep the separator in the results.

    Args:
        text (str): The text to split.
        separator (str): The separator used for splitting.
        keep_separator (bool): If True, the separator is retained in the split results.

    Returns:
        List[str]: List of split strings.
    """
    
    if separator:
        if keep_separator:
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s]


class TextSplitter:
    """
    Base class for text splitters.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size ({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Split a given text into chunks.
        
        Args:
            text (str): The text to split.

        Returns:
            List[str]: List of split strings.
        """
        pass

class RecursiveTextSplitter(TextSplitter):
    
    """
    A recursive text splitter. Splits text based on a list of separators.
    Uses a recursive approach to split text, trying various separators in sequence.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: Optional[List[str]] = None, keep_separator: bool = False, is_separator_regex: bool = False) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        """
        Initializes the RecursiveTextSplitter.

        Args:
            chunk_size (int): Desired chunk size.
            chunk_overlap (int): Overlap between chunks.
            separators (Optional[List[str]]): List of separators to use. Defaults to ["\n\n", "\n", " ", ""]
            keep_separator (bool): Whether to keep the separator in split results.
            is_separator_regex (bool): If True, separators are treated as regular expressions.
        
        """
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._keep_separator = keep_separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        
        """
        Split the provided text into chunks using the defined separators.
        
        Args:
            text (str): The text to split.

        Returns:
            List[str]: List of split strings.
        """
        
        return self._split_text(text, self._separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        
        """
        Recursive function to split the text based on a list of separators.
        
        Args:
            text (str): The text to split.
            separators (List[str]): List of separators to use for splitting.

        Returns:
            List[str]: List of split strings.
        """
        
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        _good_splits = []
        for s in splits:
            if len(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_splits = self._split_text(s, new_separators)
                    final_chunks.extend(other_splits)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        
        """
        Merge split strings based on the desired chunk size and overlap.
        
        Args:
            splits (Iterable[str]): Split strings.
            separator (str): Separator used to join split strings.

        Returns:
            List[str]: Merged strings.
        """
        
        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = len(d)
            if total + _len > self._chunk_size:
                if len(current_doc) > 0:
                    doc = separator.join(current_doc)
                    docs.append(doc)
                    while total > self._chunk_overlap:
                        total -= len(current_doc[0]) + len(separator)
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + len(separator)
        if current_doc:
            doc = separator.join(current_doc)
            docs.append(doc)
        return docs

class PDFTextSplitterWithPosition:
    """
    Extracts and splits text from a PDF with positional (page number) information.
    """
    def __init__(self, pdf_path: str, text_splitter: TextSplitter) -> None:
        """
        Initializes the PDFTextSplitterWithPosition with the path to a PDF and a TextSplitter.

        Args:
            pdf_path (str): Path to the PDF file.
            text_splitter (TextSplitter): An instance of a text splitter.
        """
        self.pdf_path = pdf_path
        self.text_splitter = text_splitter

    def extract_text_from_pdf(self) -> List[dict]:
        """
        Extracts text from a PDF and returns a list of dicts with page number and text.

        Returns:
            List[dict]: List of dicts containing page number and extracted text.
        """
        page_texts = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    page_texts.append({"page_num": page_num + 1, "text": page_text})
        return page_texts

    def split_pdf_text(self) -> List[dict]:
        """
        Splits the extracted text from a PDF into chunks using the provided TextSplitter.

        Returns:
            List[dict]: List of dicts containing page number and split text chunks.
        """
        page_texts = self.extract_text_from_pdf()
        chunks_with_position = []

        for page_info in page_texts:
            page_num = page_info["page_num"]
            text = page_info["text"]
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                chunks_with_position.append({"page_num": page_num, "text": chunk})

        return chunks_with_position