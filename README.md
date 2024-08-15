# rag-elixir-doc

- What is `RAG`: it is "chat with your documents".

- Scope: create a POC of a RAG to help/improve searching in the Elixir/Phoenix/Plug/LiveView documentation. We can focus on only "text" based documents, as these documents do not including images, or nor formatted data such as tables.

- What is our source of knowledge? We can firstly only seed the vector database with some Github raw pages from the Elixir documentation.
  We need to define how to ingest these documents to produce _embeddings_ saved into a _vector database_. Do we run a naive chunk? or [structured chunks](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_vs_recursive_retriever/), [Chunk + Document Hybrid Retrieval](https://docs.llamaindex.ai/en/stable/examples/retrievers/multi_doc_together_hybrid/), or use [BM25](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/), with an Elixir implementation [BM25](https://github.com/elliotekj/bm25)? 

- Which embedding? [SBert]()https://www.sbert.net/), or as in [this video](https://www.youtube.com/watch?v=ibzlEQmgPPY) uses "GT-SMALL" (from Alibaba).
  The problem could be to be able to use this with Elixir/`Bumblebee.Text`. The list: <https://huggingface.co/spaces/mteb/leaderboard>
  
- An index (`HNSW`) or a vector database? Postgres with PGVector, or [Supabase](https://github.com/supabase/supabase), or [ChromaDB](https://github.com/3zcurdia/chroma)??
  
- Which interface? I suggest a very simple one as this is not the focus. An input that takes a text, and a textarea where we display the response. The "context" documents should be loaded as seeds into te database. One challenge could be to format nicely <code>code</code> in the response.

- The **prompt**? This is where we define the scope of the response we want from the LLM, given the retrieved context given by the database nearest neighbour search. The LLM should be able to generate an "accurate" response constrainted by this context.

- Which **LLM**? The base LLM could be ChatGPT 3.5? Ollama? Mistral? Claude 3.5 Sonnet.  Performance vs costs.

<img width="592" alt="Screenshot 2024-08-14 at 17 56 40" src="https://github.com/user-attachments/assets/af4ef9ea-88f8-42bf-b963-013ea35d429f">

- Next? We can further improve be accepting documents, possibly links serving raw text, but also upload markdown files.

- Deploy this? As a first step, this is a POC, to be run locally.

- Source of inspiration. Repos, blog post?
  - Bumblebee, RAG: <https://hexdocs.pm/bumblebee/llms_rag.html#introduction>
  - Supabase: <https://github.com/supabase-community/chatgpt-your-files>
  - Langchain: <https://github.com/brainlid/langchain_demo>
  - <https://github.com/nileshtrivedi/autogen>

What are your thoughts?
