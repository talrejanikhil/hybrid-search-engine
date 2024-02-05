import os

from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])


def create_index(index_name):
    if index_name not in [i['name'] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=512,
            metric="dotproduct",
            spec=PodSpec(
                environment="gcp-starter"
            )
        )


def get_index(index_name):
    if index_name in [i['name'] for i in pc.list_indexes()]:
        return pc.Index(index_name)
    create_index(index_name)
    return pc.Index(index_name)
