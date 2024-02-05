from pinecone_text.sparse import BM25Encoder

from models import model, img_model
from pinecone_utils import get_index

index_name = "hybrid-search-engine"
index = get_index(index_name)
bm25 = BM25Encoder().load('bm25')


def hybrid_scale(dense, sparse, alpha):
    """Hybrid vector scaling using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: float between 0 and 1 where 0 == sparse only
               and 1 == dense only
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values': [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse


def search_products(query, query_filter, image_input=None, k=20, alpha=0.0):
    sparse_q = bm25.encode_queries(query)
    if image_input is None:
        dense_q = model.encode(query, convert_to_tensor=True, show_progress_bar=False).tolist()
    else:
        dense_q = img_model.encode(query, convert_to_tensor=True, show_progress_bar=False).tolist()

    hdense, hsparse = hybrid_scale(dense_q, sparse_q, alpha=alpha)

    search_filter = get_metadata_filter(query_filter)

    if len(hsparse['values']) > 0:
        result = index.query(
            top_k=k,
            vector=hdense,
            sparse_vector=hsparse,
            include_metadata=True,
            filter=search_filter
        )
    else:
        result = index.query(
            top_k=k,
            vector=hdense,
            include_metadata=True,
            filter=search_filter
        )
    return result['matches']


def get_metadata_filter(query_filter):
    final_filter = {}
    for index, value in query_filter.items():
        if len(value) > 0:
            final_filter[index] = {"$in": [i for i in value.split()]}
    return final_filter
