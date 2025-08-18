import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from utils.logger import logger
from utils.exception import CustomException

ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")
EMBEDDINGS_DIR = os.path.join(ARTIFACTS_DIR, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

LABEL_MAPPING = {'negative': 0, 'positive': 1}

def save_embeddings(df, features_col, output_file):
    try:
        logger.info("Starting to save embeddings...")
        texts = df[features_col].tolist()
        labels = df['rating_label'].tolist()
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

        embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        embed_df = pd.DataFrame(embeddings, columns=embedding_cols)
        embed_df['score'] = [LABEL_MAPPING[label] for label in labels]

        output_path = os.path.join(EMBEDDINGS_DIR, output_file)
        embed_df.to_csv(output_path, index=False)
        logger.info(f"Saved embeddings to {output_path}")
        return embed_df
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    from utils.config import load_csv_from_s3  
    try:
        train_df = load_csv_from_s3("train")
        test_df = load_csv_from_s3("test")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise CustomException(e, sys)

    train_embed_df = save_embeddings(train_df, "content", "train_embeddings.csv")
    test_embed_df = save_embeddings(test_df, "content", "test_embeddings.csv")
