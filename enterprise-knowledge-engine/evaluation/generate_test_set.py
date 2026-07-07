import os
import json
import uuid
import re
import structlog
from typing import List, Dict
from backend.app.services.document_processor import DocumentProcessingService
from backend.app.services.llm_generation import LLMGenerationService

logger = structlog.get_logger()

# 2. Safely initialize the synthesis engine in-memory
try:
    generation_service = LLMGenerationService()
except Exception as e:
    logger.error("Failed to spin up LLM generation service block", error=str(e))
    generation_service = None

GENERATION_PROMPT = """You are an expert QA Engineer. Given the following context block taken from an enterprise document, generate a highly specific natural language question and its corresponding definitive, factual answer based ONLY on the context.

Your output must be strict JSON matching this schema exactly with no markdown text wrapping, no code fences, and no text filler:
{{
    "question": "The specific question text...",
    "ground_truth": "The explicit factual answer text..."
}}

Context:
{context}"""

def clean_and_parse_json(raw_text: str) -> Dict:
    """
    Cleans trailing backticks, markdown code blocks, or leading text 
    to robustly parse JSON from local models.
    """
    cleaned = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return json.loads(cleaned)

def generate_synthetic_dataset(data_dir: str, output_path: str, samples_per_doc: int = 3):
    if generation_service is None:
        logger.error("Generation layer is uninitialized. Aborting synthesis loop.")
        return

    processor = DocumentProcessingService()
    test_set: List[Dict] = []
    
    if not os.path.exists(data_dir):
        logger.error("Data directory does not exist", path=data_dir)
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for file_name in os.listdir(data_dir):
        if not file_name.endswith(".pdf"):
            continue
            
        file_path = os.path.join(data_dir, file_name)
        logger.info("Generating test cases for document", filename=file_name)
        
        # Extract clean page contexts using production loaders
        records = processor.process_document(file_path, source_name=file_name)
        
        # Process a slice of parent chunks to construct distinct test variants
        chunks_to_process = records[:samples_per_doc]
        for record in chunks_to_process: 
            # Safely catch either string content configurations or explicit key bindings
            context_text = record.get("parent_text") or record.get("text") or record.get("content")
            if not context_text:
                continue
            
            try:
                target_prompt = GENERATION_PROMPT.format(context=context_text)
                
                # Use your native in-memory service instance to generate the pairs
                raw_response = generation_service.synthesize_answer(query=target_prompt, context_chunks=[])
                raw_response = raw_response.strip()
                
                # Parse structured question-answer components safely
                qa_pair = clean_and_parse_json(raw_response)
                
                # Alignment Fix: structure data to match the dynamic indexing pipeline expectations
                sample = {
                    "id": str(uuid.uuid4()),
                    "source_file": file_path,  # <-- Keeps full relative path for auto_index_test_documents
                    "context": context_text,
                    "question": qa_pair["question"],
                    "ground_truth": qa_pair["ground_truth"]
                }
                test_set.append(sample)
                logger.info("Successfully generated sample case", sample_id=sample["id"])
                
            except Exception as e:
                logger.error("Failed to generate sample for chunk", error=str(e))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=4, ensure_ascii=False)
    
    logger.info("Synthetic test dataset compiled successfully", total_samples=len(test_set), saved_to=output_path)

if __name__ == "__main__":
    generate_synthetic_dataset(data_dir="evaluation/data", output_path="evaluation/test_set.json")