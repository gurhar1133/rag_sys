### SETUP

Once you have downloaded the repo

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

if you want the query rewriting and have an openai key create a .json file called secrets.json with the following content:

```
{
    "OPENAI_API_KEY": "<your key here>"
}
```

### Run demo

To run the demo:
`python rag_sys.py`

The demo cases will follow this format:

```
YOUR QUERY: I want a book that has an epic adventure

RESULTS:

DOCUMENT: ./docs/example_doc1.txt

EVIDENCE EXCERPT: elves, dwarves, and a wizard—as they journey across vast landscapes of ancient forests, shattered ki

CHUNK ID: 4
```

### Demo docs
The example docs live in /docs as .txt files

### Demo queries
The demo queries live in /queries

### Running the demo
running `rag_sys.py` will build a vector_db from the docs, then run the queries against it with the RAG system.

### The system:
The system uses "all-MiniLM-L6-v2" and "cross-encoder/ms-marco-MiniLM-L-6-v2" for the vector store and reranker respectively. These models are lightweight and work well on a local machine with just CPU. 

The search is a hybrid of TFIDF and the vector_store. The ensamble of them currently is naive union, then reranking afterwords. But this is a good baseline and this project had it's time constraints.

Query rewriting and an initial relelvance check are handled by an openai api call, then the query is passed to the model, which will filter out responses with low similarity (cosine since this is good with text similarity). This step acts as a check to ensure that the system does not retrieve docs for which the vector store lacks relevant examples.

