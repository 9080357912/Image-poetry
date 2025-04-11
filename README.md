import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
from deep_translator import GoogleTranslator
import torch

# Load image generation pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
image_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
image_pipe = image_pipe.to(device)

# Load captioning model
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load poetry model
poetry_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
poetry_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

# Core functions
def generate_image(prompt):
    with torch.autocast("cuda" if device == "cuda" else "cpu"):
        image = image_pipe(prompt).images[0]
    return image

def generate_caption(image):
    inputs = caption_processor(images=image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_poetry(caption):
    prompt = f"Turn this into a short poem: {caption}"
    inputs = poetry_tokenizer(prompt, return_tensors="pt", max_length=100, truncation=True).to(device)
    outputs = poetry_model.generate(**inputs, max_length=100)
    poem = poetry_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return poem

def translate_poem(poem, language_code):
    return GoogleTranslator(source='auto', target=language_code).translate(poem)

def process(prompt, target_lang):
    image = generate_image(prompt)
    caption = generate_caption(image)
    poem = generate_poetry(caption)
    translated = translate_poem(poem, target_lang)
    return image, caption, poem, translated

# Gradio UI
lang_options = {
    "French": "fr",
    "Hindi": "hi",
    "Spanish": "es",
    "Arabic": "ar",
    "German": "de",
    "Japanese": "ja"
}

demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Textbox(label="Describe the image you want (prompt)", placeholder="e.g. A castle floating in the sky at sunset"),
        gr.Dropdown(choices=list(lang_options.values()), label="Translate To Language")
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Image Caption"),
        gr.Textbox(label="Poem (English)"),
        gr.Textbox(label="Poem (Translated)")
    ],
    title="🖼️🎨 Prompt-to-Poetry Multilingual Generator",
    description="Enter a prompt. It generates an image, writes a poetic caption, and translates it to your language."
)

if __name__ == "__main__":
    demo.launch()
