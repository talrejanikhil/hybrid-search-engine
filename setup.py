import os

from datasets import load_dataset
from pinecone_text.sparse import BM25Encoder
from tqdm.auto import tqdm

from models import img_model

metadata = []
fashion = []
images = []
bm25 = BM25Encoder()
index_name = "hybrid-search-engine"


def load_data():
    global fashion
    global metadata
    global images
    # load the dataset from huggingface datasets hub
    # we will only load the category "Apparel" for this project

    print('Downloading dataset')
    fashion = load_dataset(
        "ashraq/fashion-product-images-small",
        split="train"
    ).filter(lambda x: x['masterCategory'] == 'Apparel')

    metadata = fashion.remove_columns('image').to_pandas()
    images = fashion['image']

    if not os.path.exists('static'):
        os.makedirs('static')

    print('Downloading images')

    for img_id, img in list(zip(metadata['id'], images)):
        img_file_name = f'image_{img_id}.jpg'
        file_path = os.path.join('static', img_file_name)
        if not os.path.exists(file_path):
            # print(file_path)
            img.save(file_path, 'JPEG')


def fit_bm25_model():
    if 'productDisplayName' not in metadata:
        print('Run load_data first')
        return
    print('Fitting BM25 model')
    bm25.fit(metadata['productDisplayName']).dump('bm25')


def upload_data():
    if len(fashion) == 0 or len(images) == 0 or len(metadata) == 0:
        print('Run load_data first')
        return

    from pinecone_utils import get_index

    batch_size = 200
    display_record = True
    index = get_index(index_name)

    print('Indexing records to pinecone')

    for i in tqdm(range(0, len(fashion), batch_size)):
        # end of batch
        i_end = min(i + batch_size, len(fashion))
        # metadata will be upserted as sparse vectors
        meta_batch = metadata.iloc[i:i_end]
        meta_dict = meta_batch.to_dict(orient="records")
        # concatenate all metadata field except for id and year to form a single string
        meta_batch = [" ".join(x) for x in meta_batch.loc[:, ~meta_batch.columns.isin(['id', 'year'])].values.tolist()]
        if display_record:
            print(f'\nMetadata record: {meta_batch[0]}')
        # images will be upserted as dense vectors
        img_batch = images[i:i_end]
        dense_embeds = img_model.encode(img_batch).tolist()
        # create sparse BM25 vectors
        sparse_embeds = bm25.encode_documents([text for text in meta_batch])
        # create unique IDs
        ids = [str(x) for x in range(i, i_end)]
        upserts = []
        # loop through the data and create dictionaries for uploading documents to pinecone index
        for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, meta_dict):
            # We display one record that will be upserted as an example
            if display_record:
                print(
                    f'Upserting record with id: {_id},\nsparse_vector:\n {sparse}, \ndense_vector:\n {dense} '
                    f'\nand metadata:\n {meta}')
                display_record = False
            upserts.append({
                'id': _id,
                'values': dense,
                'sparse_values': sparse,
                'metadata': meta
            })
        index.upsert(upserts)
    print(index.describe_index_stats())


if __name__ == "__main__":
    load_data()
    fit_bm25_model()
    upload_data()
