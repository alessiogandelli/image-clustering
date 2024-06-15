#%%
import pandas as pd
import base64
import requests
from dotenv import load_dotenv
import os
from bertopic import BERTopic
from bertopic.representation import VisualRepresentation
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import json
import matplotlib.pyplot as plt

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = SentenceTransformer('clip-ViT-B-32')
model_text = SentenceTransformer('all-MiniLM-L6-v2')


# %%


img_path = '/Users/alessiogandelli/dev/uni/image-clustering/data/imgs'
img_files = [os.path.join(img_path, filename) for filename in os.listdir(img_path)]



def get_img_embeddings(img_files):
    images_to_embed = [Image.open(filepath) for filepath in img_files]
    img_emb = model.encode(images_to_embed, show_progress_bar=True)

    return img_emb, images_to_embed


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

img_emb, images = get_img_embeddings(img_files)

#%%



reduced_embeddings = UMAP(n_components=5).fit_transform(img_emb)

clustering =  HDBSCAN(
    min_cluster_size=2,  # Lowered to accommodate small clusters
    min_samples=2,  # Close to min_cluster_size for consistency
    cluster_selection_method='eom',  # Excess of Mass for flexibility
    prediction_data=True  # Allows for soft clustering and outlier scores
).fit(reduced_embeddings)


#%%



# %%

# OpenAI API Key

# Function to encode the image
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

prompt = """I have an image, and I would like to perform the following tasks:

        1. Generate a Caption: Provide a brief description of what is depicted in the image.
        2. Extract Text: Identify and extract any text present within the image.
        3. Generate Tags: Provide 3 relevant tags or keywords that describe the image content.

        format the answer as a json object with the following structure:
        {
            "caption": "A brief description of the image content.",
            "text": "Any text identified within the image.",
            "tags": ["tag1", "tag2", "tag3"]
        }
        """

labeled_images = []


for i in range(len(img_files)):
   

    image_path = img_files[i]

    base64_image = encode_image(image_path) # Getting the base64 string



    payload = {
    "model": "gpt-4o",
    "response_format": { "type": "json_object" },
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
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json()['choices'][0]['message']['content'])

    parsed_response = response.json()['choices'][0]['message']['content']
    json_string = json.loads(parsed_response)
    json_string['embedding'] = img_emb[i].tolist()
    json_string['image_path'] = image_path
    labeled_images.append(json_string)



#%%
# %%

#save the labeled images
with open('labeled_images.json', 'w') as f:
    json.dump(labeled_images, f)

# %%

emb2d  = UMAP(n_components=2).fit_transform(img_emb)


plt.figure(figsize=(15, 15))
plt.scatter(emb2d[:, 0], emb2d[:, 1])
for i, label in enumerate(labeled_images):
    plt.annotate(label['tags'], (emb2d[i, 0], emb2d[i, 1]))
plt.show()

# %%
import matplotlib.pyplot as plt
from PIL import Image

# Assuming emb2d is your 2D embeddings array, labeled_images contains labels, and images is a list of image paths

# %%
