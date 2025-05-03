# RAG - Article Query Filter

This project implements a Retrieval Augmented Generation (RAG) index to efficiently query articles and retrieve the most relevant content while filtering out harmful or toxic queries.

### 0. Data analysis

Code in **00_eda.ipynb**

Exploratory data analysis of corpus.csv and queries.csv datasets.

### 1. Data preparation

Code in **01_data_preparation.py**

Data preparation steps:
- Concatenated title and corpus text
- Text normalization (removed unicode, extra spaces, lowercase)

#### Next steps & future improvements:
- Unit and functional tests
- Dataset versioning with [DVC](https://dvc.org/doc) 
- Named Entity Recognition (NER) to extract entities from text

### 2. RAG index

#### What is RAG indexing?

RAG indexing is a process designed to optimize data retrieval for Retrieval-Augmented Generation (RAG) systems. Key components:
- **Data Loading**: Collect raw data from various sources (e.g., documents, APIs) and transform it into structured Document objects, which include text and metadata.
- **Chunking**: Split the documents into smaller segments, such as sentences or paragraphs. This improves the performance of RAG models by allowing them to handle more focused pieces of information.
- **Embedding**: Convert each text chunk into vector representations using embedding techniques. These vectors capture the semantic meaning of the text, facilitating efficient similarity searches.
- **Indexing**: Store the generated vectors in a vector database in an indexed format, enabling quick retrieval based on user queries.
- **Query Processing**: When a query is made, it is also embedded into a vector, which is then compared against the indexed vectors to find relevant information.

![rag_index](https://github.com/user-attachments/assets/7d8adf8e-957d-4e12-8e9a-cb16425fed41)


Code in **02_rag_index.ipynb**

Created a vector store index from corpus using framework [llamaindex](https://www.llamaindex.ai/) and [ChromaDB](https://docs.trychroma.com/) as vectorDB.

Evaluation on 500 queries sample --> 27% Recall (matched documents / total relevant documents)

#### Next steps & future improvements:
- Modular Python code
- Integration tests
- [Semantic chunking](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/)
- Improve retrieval evaluation with additional samples and metrics
- Extend to multilingual document retrieval
- Store vector db remotely

### 3. Toxic classifier

Code in **03_toxic_classifier.ipynb**

Since queries dataset have no labels (toxic vs non-toxic) and the maximum number of tokens for queries is 94, a BERT pre-trained model has been used as toxic classifier.
In particular, it is a version of the bert-base-uncased model fine-tuned on Jigsaw Toxic Dataset, available in Hugging Face.

References:
- https://huggingface.co/JungleLee/bert-toxic-comment-classification
- https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data

### 4. API

Code in **04_api.py**

Built a POST API using [FastAPI](https://fastapi.tiangolo.com/).

Requirements: If the query is non-toxic, the related uuids are returned as response.

#### How does it work?

1. Run local server
```
uvicorn 04_api:app
```
2. Visit FastAPI Swagger UI: http://127.0.0.1:8000/docs
3. Click on POST API selection and "Try it out" button
4. Test you query
   
![api_demo](https://github.com/user-attachments/assets/9d471bb1-c44e-4313-8674-90fd072d6483)

#### Next steps & future improvements:
- Unit and integration tests
- Optimize response time

