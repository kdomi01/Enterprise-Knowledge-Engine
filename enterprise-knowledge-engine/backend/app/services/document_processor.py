import uuid
from typing import Dict, List, Tuple
import pdfplumber
import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

logger = structlog.get_logger()

def clean_transcript_layout(text: str) -> str:
    """
    Scans extracted layout text to stitch broken multi-line subject strings
    back into unified semantic rows.
    """
    lines = text.split("\n")
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        # If a line starts with huge whitespace or trailing lowercase fragments, 
        # it's likely a wrapped string from the row above it.
        if i > 0 and line.startswith("   ") and len(line.strip()) > 0 and not re.search(r'\b(Pass|Fail|\d+)\b', line):
            # Stitch it to the previous line
            cleaned_lines[-1] = cleaned_lines[-1].strip() + " " + line.strip()
        else:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)

class DocumentProcessingService:
    def __init__(self):
        # 1. Define the large parent chunk layout
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # 2. Define the crisp child chunk layout
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=40,
            length_function=len,
        )

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extracts raw text and runs layout-stitching on the entire document.
        """
        full_text = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(layout=True)
                if page_text:
                    # Apply stitching cleanup page-by-page
                    cleaned_page = clean_transcript_layout(page_text)
                    full_text.append(cleaned_page)
                
        return "\n--- Page Break ---\n".join(full_text)

    def process_document(self, file_path: str, source_name: str) -> List[Dict]:
        """
        Document-agnostic ingestion pipeline. Uses page-level layout separation
        and text restructuring to preserve data grids, tables, and multi-line strings.
        """
        processed_records = []
        
        # 1. Extract text page-by-page to respect physical document boundaries
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info("Parsing document layout", filename=source_name, total_pages=total_pages)
            
            # Extract and stitch layout text for each page separately
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text(layout=True)
                if text and text.strip():
                    # Apply text row stitching preprocessing here!
                    cleaned_text = clean_transcript_layout(text)
                    pages_text.append(cleaned_text.strip())

        if not pages_text:
            logger.warning("No readable text extracted from document", filename=source_name)
            return []

        # 2. DYNAMIC STRATEGY ALLOCATION
        # Case A: Short or Dense Tabular Assets (Transcripts, Certificates, Invoices)
        if total_pages <= 3:
            logger.info("Processing as short/dense unified context layout.", filename=source_name)
            full_context = "\n\n--- Page Break ---\n\n".join(pages_text)
            unified_id = str(uuid.uuid4())
            
            return [{
                "child_id": str(uuid.uuid4()),
                "child_text": full_context,
                "parent_id": unified_id,
                "parent_text": full_context,
                "metadata": {
                    "source": source_name,
                    "page_start": 1,
                    "page_end": total_pages
                }
            }]

        # Case B: Multi-page Narrative Documents (Manuals, Guidelines, Reports)
        logger.info("Processing as multi-page segmented context layout.", filename=source_name)
        for idx, page_content in enumerate(pages_text):
            page_num = idx + 1
            parent_id = str(uuid.uuid4())
            
            # Each page serves as its own Parent chunk to keep context localized
            record = {
                "child_id": str(uuid.uuid4()),
                "child_text": page_content,
                "parent_id": parent_id,
                "parent_text": page_content,
                "metadata": {
                    "source": source_name,
                    "page_start": page_num,
                    "page_end": page_num
                }
            }
            processed_records.append(record)

        logger.info("Layout-aware ingestion compiled successfully", chunks_generated=len(processed_records))
        return processed_records