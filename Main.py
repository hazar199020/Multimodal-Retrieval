from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from langchain_core.documents import Document
import torch
import torch.nn.functional as F
import os

# Wrap CLIP in a LangChain-compatible embedding class
class CLIPEmbeddings(Embeddings):

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    #indexing/storing content (text documents)
    def embed_documents(self, texts):
        inputs =  self.processor(text=texts, return_tensors="pt", padding=True)
        #embedding will be torch.tensor with 512 length
        embs = self.model.get_text_features(**inputs)
        #To make retrieval more stable, normalize vectors before returning:
        embs = F.normalize(embs, p=2, dim=-1)
        #Because FAISS and most vector DBs expect NumPy.
        embs = embs.detach().cpu().numpy()
        return embs

    #searching/retrieving (text query)
    def embed_query(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        embs = self.model.get_text_features(**inputs)
        embs = F.normalize(embs, p=2, dim=-1)
        return embs[0].detach().cpu().numpy()

    #The image embedding encodes semantic features (shapes, colors, objects, context)
    def embed_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        embs = self.model.get_image_features(**inputs)
        embs = F.normalize(embs, p=2, dim=-1)
        return embs.detach().cpu().numpy().squeeze()

def main():
    # Disable the warning
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Load CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


    # Initialize vector store
    clip_embeddings = CLIPEmbeddings(clip_model,clip_processor)

    #add Text embedding
    vectorstore = FAISS.from_texts(["cat", "dog", "camel", "laptop"], embedding=clip_embeddings)

    #add Image embedding
    image_paths = ["Images/cat.jpg", "Images/dog.jpg", "Images/sofa.jpg"]
    docs = []
    embs = []

    for path in image_paths:
      image = Image.open(path)
      emb = clip_embeddings.embed_image(image)
      embs.append(emb)
      docs.append(Document(page_content=path, metadata={"type": "image"}))

    # Now add to FAISS
    # Step 2: add image embeddings directly
    # add document content (here it is image path) First and pass it as ZIP of tuple
    vectorstore.add_embeddings([(doc.page_content, emb) for doc, emb in zip(docs, embs)])

    print("The number of items in KB", vectorstore.index.ntotal)

    #Text Query
    query = "animal"
    docs = vectorstore.similarity_search(query, k=3)
    print("Query is "+query+" Answer is " + docs[0].page_content)

    query_vector = clip_embeddings.embed_query(query)
    # Compare manually with stored vectors
    for doc_vector in vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal):
      cosine = np.dot(query_vector, doc_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
      )
      print("Cosine similarity: ", cosine, "  ", )

     #Image Query
    image = Image.open("Images/camel.jpg")
    image_emb = clip_embeddings.embed_image(image)
    docs = vectorstore.similarity_search_by_vector(image_emb, k=1)
    print("Query is "+image.filename+" Answer is ", docs[0].page_content)

    #query_vector = clip_embeddings.embed_query(query)
    query_vector = image_emb
    # Compare manually with stored vectors
    for doc_vector in vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal):
         cosine = np.dot(query_vector, doc_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
         )
         print("Cosine similarity:", cosine, "  ", )

if __name__ == '__main__':
    main()