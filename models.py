from sentence_transformers import SentenceTransformer

# Here we load the multilingual CLIP model. Note, this model can only encode text.
model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
# If you need embeddings for images, you must load the 'clip-ViT-B-32' model
img_model = SentenceTransformer('clip-ViT-B-32')
