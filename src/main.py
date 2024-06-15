#%%
import pandas as pd
import base64
import requests
from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv("SUPER_SECRET_KEY"))


# %%
from bertopic import BERTopic
from bertopic.representation import VisualRepresentation
from sentence_transformers import SentenceTransformer, util
from PIL import Image


img_path = '/Users/alessiogandelli/dev/uni/image-clustering/data/imgs'
img_files = [os.path.join(img_path, filename) for filename in os.listdir(img_path)]

model = SentenceTransformer('clip-ViT-B-32')

images_to_embed = [Image.open(filepath) for filepath in img_files]
img_emb = model.encode(images_to_embed, show_progress_bar=True)

for image in images_to_embed:
    image.close()
# %%





# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = img_files[0]

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

prompt = """I have an image, and I would like to perform the following tasks:

        1. Generate a Caption: Provide a brief description of what is depicted in the image.
        2. Extract Text: Identify and extract any text present within the image.
        3. Generate Tags: Provide relevant tags or keywords that describe the image content.

        format the answer as a json object with the following structure:
        {
            "caption": "A brief description of the image content.",
            "text": "Any text identified within the image.",
            "tags": ["tag1", "tag2", "tag3"]
        }
"""



payload = {
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}


#%%
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json()['choices'][0]['message']['content'])
# %%
