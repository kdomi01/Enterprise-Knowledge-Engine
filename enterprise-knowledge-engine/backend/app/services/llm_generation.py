import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import structlog

logger = structlog.get_logger()

class LLMGenerationService:
    def __init__(self):
        # We use a highly efficient, instruct-tuned local model
        self.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info("Initializing local LLM Synthesis Engine...", model=self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Local LLM Synthesis Engine loaded successfully", device=self.device)
        except Exception as e:
            logger.error("Failed to initialize local LLM engine", error=str(e))
            self.generator = None

    def synthesize_answer(self, query: str, context_chunks: list) -> str:
        """
        Grounded, document-agnostic answer synthesis. Handles any document type
        (manuals, specs, contracts, text) by strictly tying generation to retrieved context.
        """
        if not self.generator:
            return "Generation failed: LLM engine uninitialized."

        # Compile context from any source document type cleanly
        context_str = "\n\n".join([f"[Context Source: {c.get('source', 'Unknown')}]: {c.get('text', '')}" for c in context_chunks])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise, context-grounded document analysis engine. Your task is to answer the user's query using only the provided context snippets." 
                    "STRICT OPERATIONAL DIRECTIVES:" 
                    "1. Grounding: Rely strictly on facts directly mentioned in the context. Do not bring in unverified external knowledge."
                    "2. Tabular Reconstruction: If data tables, schemas, or forms are split across multi-line breaks (e.g., subject names or keys broken across lines), you are explicitly required to structurally reconstruct them cleanly for the user."
                    "3. Analytical Calculations: If the context provides explicit mathematical formulas AND the raw numerical inputs needed for those formulas are present in the text, you are authorized and expected to perform step-by-step arithmetic to calculate the exact answer. Show your math clearly."
                    "4. Missing Data Fallback: If required fields or calculation parameters are fundamentally absent from the text, state clearly what specific data point is missing rather than generalizing."
                )
            },
            {
                "role": "user",
                "content": f"Retrieved Context:\n{context_str}\n\nUser Question: {query}\n\nAnswer:"
            }
        ]

        try:
            outputs = self.generator(
                messages,
                max_new_tokens=1024,  # Room for complex technical or regulatory explanations
                temperature=0.1,     # Keeps responses highly deterministic and anchored
                top_p=0.9
            )
            return outputs[0]["generated_text"][-1]["content"].strip()
        except Exception as e:
            logger.error("Error during universal text synthesis pass", error=str(e))
            return f"Error executing generation pass: {str(e)}"
        
        try:
            # Generate the response using optimal generation parameters
            outputs = self.generator(
                messages,
                max_new_tokens=1024,
                temperature=0.1,  # Low temperature forces deterministic, grounded output
                top_p=0.9
            )
            # Pull text from the ChatTemplate structure
            generated_text = outputs[0]["generated_text"][-1]["content"]
            return generated_text.strip()
        except Exception as e:
            logger.error("Error during text generation pass", error=str(e))
            return f"Error executing generation pass: {str(e)}"