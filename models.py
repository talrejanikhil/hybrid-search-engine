# Import the SentenceTransformer class from the sentence_transformers library.
# This class is designed to work with sentence embeddings, leveraging pre-trained models for various NLP tasks.
from sentence_transformers import SentenceTransformer

# Initialize a SentenceTransformer model with the 'clip-ViT-B-32-multilingual-v1' identifier.
# This particular model is a multilingual version of the CLIP model, specifically designed to encode text inputs.
# It supports multiple languages, making it versatile for international applications.
# Note: This model is optimized for text encoding and does not support image encoding.
model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

# To work with images, we load a different CLIP model using the SentenceTransformer class.
# The 'clip-ViT-B-32' model is designed for both text and image inputs, but in this instance,
# it's intended for image encoding.
# This model leverages the Vision Transformer (ViT) architecture with a B/32 configuration,
# suitable for a wide range of image encoding tasks.
img_model = SentenceTransformer('clip-ViT-B-32')
