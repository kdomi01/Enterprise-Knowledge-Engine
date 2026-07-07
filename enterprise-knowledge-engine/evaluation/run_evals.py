import json
import os
import re
import structlog
from typing import List, Dict
from backend.app.graph.workflow import rag_graph
from evaluation.index_eval_data import auto_index_test_documents

# 1. Import your production service class instead of a loose 'llm' variable
from backend.app.services.llm_generation import LLMGenerationService

logger = structlog.get_logger()

# 2. Safely initialize your local synthesis engine in-memory
try:
    judge_llm_service = LLMGenerationService()
except Exception as e:
    logger.error("Failed to spin up Judge LLM service block", error=str(e))
    judge_llm_service = None

JUDGE_PROMPT = """You are an independent Academic Auditor. Evaluate the provided text layers based strictly on these rules:
1. Faithfulness (0.0 - 1.0): Is the generation completely derived from the retrieved context without introducing hallucinations?
2. Context Precision (0.0 - 1.0): Did the system retrieve relevant text blocks capable of answering the question?

Respond in strict JSON matching this schema format exactly with no markdown text wrapping, no code fences, and no text filler:
{{
    "faithfulness": 0.9,
    "context_precision": 1.0,
    "critique": "Brief explanation of the score"
}}

Question: {question}
Retrieved Context: {context}
Generated Answer: {generation}"""

def call_judge_llm(prompt: str) -> Dict:
    """
    Queries the local judge LLM via the native LLMGenerationService wrapper
    and returns the parsed scoring metrics.
    """
    if judge_llm_service is None:
        logger.error("Judge LLM service layer is uninitialized.")
        return {"faithfulness": 0.0, "context_precision": 0.0, "critique": "Judge service offline."}
        
    try:
        # 3. Leverage your native model configuration via synthesize_answer
        raw_text = judge_llm_service.synthesize_answer(query=prompt, context_chunks=[])
        raw_text = raw_text.strip()
        
        # Defensive clean up for markdown fences before json parsing
        cleaned = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        
        return json.loads(cleaned)
    except Exception as e:
        logger.error("Judge evaluation processing failed, defaulting to zero-scores", error=str(e))
        return {"faithfulness": 0.0, "context_precision": 0.0, "critique": f"Failed to audit: {str(e)}"}
    
def run_evaluation_suite(test_set_path: str, report_path: str):
    # 1. Verify path existence first
    if not os.path.exists(test_set_path):
        logger.error(f"Test set missing at target location: {test_set_path}")
        return

    # 2. Dynamically clear db and index files defined in your test set
    logger.info("Initializing automated pipeline document ingestion...")
    auto_index_test_documents(test_set_path)
    
    # 3. Load the test set file into memory safely before using it
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    evaluation_results = []
    total_faithfulness = 0.0
    total_precision = 0.0

    logger.info("Starting evaluation run", total_queries=len(test_set))

    for idx, test_case in enumerate(test_set):
        logger.info("Evaluating sample query", progress=f"{idx+1}/{len(test_set)}")
        
        # Fire the query straight into your production LangGraph orchestrator
        initial_state = {
            "query": test_case["question"],
            "retrieved_documents": [],
            "generation": "",
            "steps": []
        }
        
        try:
            final_state = rag_graph.invoke(initial_state)
            
            generation = final_state.get("generation", "")
            
            # Account for either object schemas or primitive dictionaries from the graph state context
            retrieved_docs = final_state.get("reranked_documents", final_state.get("retrieved_documents", []))
            retrieved_contexts = [
                doc.text if hasattr(doc, 'text') else doc.get("text", "") 
                for doc in retrieved_docs
            ]
            unified_context = "\n\n".join(retrieved_contexts)

            # Package and evaluate via LLM-as-a-Judge
            judge_prompt_formatted = JUDGE_PROMPT.format(
                question=test_case["question"],
                context=unified_context if unified_context.strip() else "NO CONTEXT RETRIEVED",
                generation=generation if generation.strip() else "NO GENERATION RETURNED"
            )
            
            audit_metrics = call_judge_llm(judge_prompt_formatted)
            
            # Compile metric records
            eval_metrics = {
                "query": test_case["question"],
                "ground_truth": test_case["ground_truth"],
                "system_generation": generation,
                "visited_steps": final_state.get("steps", []),
                "scores": {
                    "faithfulness": float(audit_metrics.get("faithfulness", 0.0)),
                    "context_precision": float(audit_metrics.get("context_precision", 0.0))
                },
                "critique": audit_metrics.get("critique", "No critique provided.")
            }
            
            total_faithfulness += eval_metrics["scores"]["faithfulness"]
            total_precision += eval_metrics["scores"]["context_precision"]
            evaluation_results.append(eval_metrics)
            
        except Exception as pipeline_error:
            logger.error("Failed to run pipeline evaluation on sample", error=str(pipeline_error))
            continue

    # Compute macro averages
    num_cases = len(evaluation_results)
    summary_report = {
        "metrics_summary": {
            "avg_faithfulness": round(total_faithfulness / num_cases, 4) if num_cases > 0 else 0.0,
            "avg_context_precision": round(total_precision / num_cases, 4) if num_cases > 0 else 0.0,
            "total_evaluated_cases": num_cases
        },
        "detailed_runs": evaluation_results
    }

    # Ensure output destination target folder tracks exist
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4, ensure_ascii=False)

    logger.info("Evaluation suite complete", report_saved_to=report_path)

if __name__ == "__main__":
    run_evaluation_suite(test_set_path="evaluation/test_set.json", report_path="evaluation/eval_report.json")