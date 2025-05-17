import os
import argparse
import pandas as pd
import numpy as np
from speaker_clustering.embedding import extract_embedding
from speaker_clustering.clustering import cluster_embeddings


def main():
    parser = argparse.ArgumentParser(description="Speaker Clustering using ECAPA-TDNN")
    parser.add_argument(
        "audio_folder", type=str, help="Path to folder containing audio files"
    )
    parser.add_argument(
        "--use_cosine", action="store_true", help="Use cosine distance (default: False)"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters (if not using threshold)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Distance threshold for clustering"
    )
    parser.add_argument(
        "--output", type=str, default="speaker_clusters.csv", help="Output CSV filename"
    )
    args = parser.parse_args()

    audio_files = []
    embeddings = []

    for root, _, files in os.walk(args.audio_folder):
        for file in files:
            if file.endswith((".wav", ".mp3")):
                path = os.path.join(root, file)
                emb = extract_embedding(path)
                if emb is not None:
                    audio_files.append(file)
                    embeddings.append(emb)

    if not embeddings:
        print("هیچ فایل صوتی معتبری پیدا نشد.")
        return

    embeddings = np.array(embeddings)
    labels = cluster_embeddings(
        embeddings,
        use_cosine=args.use_cosine,
        n_clusters=args.n_clusters,
        threshold=args.threshold,
    )

    df = pd.DataFrame({"filename": audio_files, "speaker_id": labels})
    df.to_csv(args.output, index=False)
    print(f"نتایج در {args.output} ذخیره شد.")


if __name__ == "__main__":
    main()
