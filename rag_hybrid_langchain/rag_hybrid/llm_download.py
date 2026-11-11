from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-4B", device_map="auto")