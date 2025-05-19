from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from crawling_loader import load_reviews

tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")
model = AutoModel.from_pretrained("jhgan/ko-sroberta-multitask")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings[0].numpy()

def main():
    # 1. 리뷰 데이터 불러오기
    review_texts = load_reviews('reviews.csv')

    # 2. 임베딩 생성
    print("임베딩 생성 중...")
    embeddings = [get_embedding(text) for text in review_texts]

    # 3. Chroma DB에 저장
    print("Chroma DB에 저장 중...")
    client = chromadb.Client()
    collection = client.create_collection("reviews")
    for i, (text, vector) in enumerate(zip(review_texts, embeddings)):
        collection.add(
            ids=[str(i)],
            embeddings=[vector.tolist()],
            documents=[text]
        )
    print(f"총 {len(review_texts)}건 저장 완료!")

if __name__ == "__main__":
    main() 