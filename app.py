import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_pipeline():
    try:
        # Model name
        model_name = "openbmb/MiniCPM-Llama3-V-2_5"
        logger.info(f"Initializing model: {model_name}")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("Tokenizer initialized successfully.")

        # Initialize model with disk offloading
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            logger.info("Model loaded into memory with low CPU usage.")

        # Infer device map and dispatch model
        device_map = infer_auto_device_map(model, no_split_module_classes=["AutoModelForCausalLM"])
        dispatch_model(model, device_map=device_map, offload_folder="model_offload")
        logger.info(f"Model dispatched with device map: {device_map}")

        # Create and return the pipeline
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
        )
        logger.info("Pipeline initialized successfully.")

        return text_pipeline

    except Exception as e:
        logger.error(f"An error occurred during pipeline initialization: {e}")
        raise


if __name__ == "__main__":
    # Initialize the pipeline
    logger.info("Starting the pipeline initialization...")
    pipeline_instance = initialize_pipeline()

    # Example usage
    prompt = "Describe the significance of machine learning in healthcare."
    logger.info(f"Input prompt: {prompt}")
    output = pipeline_instance(prompt, max_length=100, num_return_sequences=1)
    logger.info(f"Generated text: {output[0]['generated_text']}")

