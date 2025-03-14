from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load a local image
image_path = "./dogtest.png"  # Local file path
image = Image.open(image_path).convert("RGB")

# Process the image for captioning
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)

# Output the generated caption
caption = processor.batch_decode(out, skip_special_tokens=True)[0]
print("Generated Caption:", caption)
