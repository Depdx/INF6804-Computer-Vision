import skimage
from src.database import Database
from src.feature_extractors.local_binary_pattern import LocalBinaryPattern
from src.descriptors.histogram import Histogram
from src.similarities.cosine_similarity import CosineSimilarity

db = Database(
    directory="./data/database",
    feature_extractor=LocalBinaryPattern(
        n_points=20,
        radius=10,
        method="uniform",
    ),
    descriptor=Histogram(),
)

image_path = "./data/cat_query.jpg"
query = skimage.io.imread(image_path)

results = db.query(
    query=query,
    similarity_measure=CosineSimilarity(),
    k=10,
)

for similarity, image, features, file in results:
    print(f"Similarity: {similarity}")
    print(f"File: {file}")
    skimage.io.imshow(image)
    skimage.io.show()


# grid search
# implementation couleur pour co-occurence matrix

# Autmatic topk-accuracy

# add timer (generate database, query, timer de toutes les étapes)
# add wandb
# add visualisation des résultats
# report
