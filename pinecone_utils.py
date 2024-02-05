import os

# Import Pinecone and PodSpec from the pinecone module.
from pinecone import Pinecone, PodSpec

# Initialize a Pinecone client with an API key.
# The API key is read from the environment variables
# This client will be used to interact with Pinecone services, such as creating and listing indexes.
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])


# Define a function to create a new index in Pinecone if it doesn't already exist.
def create_index(index_name):
    # Check if the index already exists by comparing the provided index_name with the list of existing indexes.
    if index_name not in [i['name'] for i in pc.list_indexes()]:
        # If the index does not exist, create a new one using the specified parameters.
        pc.create_index(
            name=index_name,  # Name of the index
            dimension=512,  # The dimension of the vectors that will be stored in the index
            metric="dotproduct",  # The similarity metric to be used for vector comparison
            spec=PodSpec(
                environment="gcp-starter"
            )
        )


# Define a function to get an index by name.
# This function ensures an index is available for use, creating one if necessary.
def get_index(index_name):
    # Check if the index exists among the list of indexes.
    if index_name in [i['name'] for i in pc.list_indexes()]:
        # If the index exists, return a Pinecone Index object for interacting with the index.
        return pc.Index(index_name)
    # If the index does not exist, create it first.
    create_index(index_name)
    # After ensuring the index is created, return the Pinecone Index object.
    return pc.Index(index_name)
