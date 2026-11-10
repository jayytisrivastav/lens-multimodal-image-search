from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

project = Path(__file__).resolve().parent
images_folder = project / "images"

if not images_folder.is_dir():
    raise SystemExit("Create a folder named images beside build_index.py.")

photos = sorted(
    path for path in images_folder.iterdir()
    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)
if not photos:
    raise SystemExit("Copy at least one JPG or PNG into the images folder.")

print(f"Found {len(photos)} image(s). Loading Jina CLIP on CPU...")
model = SentenceTransformer(
    "jinaai/jina-clip-v2",
    trust_remote_code=True,
    truncate_dim=1024,
    device="cpu",
)

client = QdrantClient(path=str(project / "image_store"))
try:
    if not client.collection_exists("images"):
        client.create_collection(
            collection_name="images",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    for photo in photos:
        with Image.open(photo) as image:
            vector = model.encode(
                [image.convert("RGB")], normalize_embeddings=True
            )[0]

        relative_path = photo.relative_to(project).as_posix()
        point = PointStruct(
            id=str(uuid5(NAMESPACE_URL, relative_path)),
            vector=vector.tolist(),
            payload={"path": relative_path},
        )
        client.upsert(collection_name="images", points=[point], wait=True)
        print(f"Indexed {photo.name}")

    total = client.count(collection_name="images", exact=True).count
    print(f"Done. Images in the index: {total}")
finally:
    client.close()