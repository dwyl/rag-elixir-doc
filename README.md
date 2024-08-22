# rag-elixir-doc

- What is `RAG`: it is "chat with your documents", meaning you ask an LLM model to respond based on additional sources. Theses sources may or may not be already incorporated inside the training used for the LLM. Using RAG is _not about fine tuning_ the model, which is changing the coefficients or structure of the model based on additional sources. RAG is about giving an additional context - the "context window" - to enhance or constraint the response from the LLM. Note that the LLM accepts a limited amounts of tokens, thus the window context is limited.

- Scope: create a POC of a RAG to help/improve searching in the Elixir/Phoenix/Plug/LiveView documentation. We can focus on only "text" based documents, as these documents do not including images, or nor formatted data such as tables.

- What is our source of knowledge? We firstly seed the vector database with some Github markdown pages from the Elixir documentation.

  The sources will be extracted from the files that the GitHub API returns when querying some directories:
    - <https://api.github.com/repos/phoenixframework/phoenix_live_view/contents/guides/server/>
    - <https://api.github.com/repos/phoenixframework/phoenix_live_view/contents/guides/client/>
    - <https://github.com/phoenixframework/phoenix_live_view/blob/main/guides/introduction/welcome.md>

    - we can also add some ".ex" modules when they provide documentation in a moduledoc.


- Overview of the process:
  
  * Build the external sources.
    - Download "external sources" as a string
    - chunk the sources
    - produce an embedding based on a "sentence-transformer" model for each chunk
    - insert chunk + embedding into a Vector database
    - build an index
  * Build a prompt that wraps your question with a semantically close context extracted from the additional sources
    - produce an embedding from the question
    - first vector similarity search (HNSW)
    - rerank the top-k with "cross-encoding"
    - inject the later result with the query into a prompt
    - submit the prompt to the LLM

  * in pseudo-code, the pipeline will be:
  
```elixir
# Data collection and chunking
defmodule DataCollector do
  def fetch_and_chunk_docs do
    ...
  end
end

# Embedding generation: "sentence-transformers/all-MiniLM-L6-v2"
defmodule Embedder do
  def generate_embeddings(text) do
    ...
  end
end

# Semantic search
defmodule SemanticSearch do
  def search(query, top_k) do
    ...
  end
end

# Cross-encoder reranking: "cross-encoder/ms-marco-MiniLM-L-6-v2"
defmodule CrossEncoder do
  def rerank(query, documents) do
    ...
  end
end

# Prompt construction
defmodule PromptBuilder do
  def build_prompt(query, context) do
    ...
  end
end

# LLM integration
defmodule LLM do
  def generate_response(prompt) do
    ...
  end
end

# Main RAG pipeline
defmodule RAG do
  def process_query(query) do
    query
    |> SemanticSearch.search(10)
    |> CrossEncoder.rerank(query)
    |> PromptBuilder.build_prompt(query)
    |> LLM.generate_response()
  end
end
```
- What is **bi-encoding** and **cross-encoding**?
  - Bi-encoders: Encode the query and document separately, then compare their vector representations.
  - Cross-encoders: Take both the query and document as input simultaneously, allowing for more complex interactions between them.
  - How cross-encoders works in reranking?:
    - After initial retrieval (e.g., using vector similarity), you pass each query-document pair through the cross-encoder.
    - The cross-encoder outputs a relevance score for each pair.
    - Results are then sorted based on these scores, potentially significantly changing the order from the initial retrieval.
    
  
- how to **chunk**? We need to define how to ingest these documents to produce _embeddings_ saved into a _vector database_. Do we run a naive chunk? or [use this package](https://github.com/revelrylabs/text_chunker_ex), or [structured chunks](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_vs_recursive_retriever/), [Chunk + Document Hybrid Retrieval](https://docs.llamaindex.ai/en/stable/examples/retrievers/multi_doc_together_hybrid/), or use [BM25](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/), with an Elixir implementation [BM25](https://github.com/elliotekj/bm25)? 

- Which embedding? [SBert]()https://www.sbert.net/), or as in [this video](https://www.youtube.com/watch?v=ibzlEQmgPPY) uses "GT-SMALL" (from Alibaba) or "sentence-transformer". Check: <https://huggingface.co/spaces/mteb/leaderboard>. We will use "sentence-transformer".
  
- An index (`HNSW`) or a vector database? Postgres with PGVector, or [Supabase](https://github.com/supabase/supabase), or [ChromaDB](https://github.com/3zcurdia/chroma)? We will use Postgres with the extension PG_Vector and the HNSWL algorithm.
  
- The **prompt**? This is where we define the scope of the response we want from the LLM, given the retrieved context given by the database nearest neighbour search. The LLM should be able to generate an "accurate" response constrainted by this context.

- Which **LLM**? The base LLM could be ChatGPT 3.5? Ollama? Mistral? Claude 3.5 Sonnet. We will choose to use `Ollama` since you can install and run it locally.

<img width="592" alt="Screenshot 2024-08-14 at 17 56 40" src="https://github.com/user-attachments/assets/af4ef9ea-88f8-42bf-b963-013ea35d429f">

- Next? We can further improve be accepting documents, possibly links serving raw text, but also upload markdown files.

- Deploy this? As a first step, this is a POC, to be run locally.

- Source of inspiration. Repos, blog post?
  - Bumblebee, RAG: <https://hexdocs.pm/bumblebee/llms_rag.html#introduction>
  - Supabase: <https://github.com/supabase-community/chatgpt-your-files>
  - Langchain: <https://github.com/brainlid/langchain_demo>
  - <https://github.com/nileshtrivedi/autogen>
  - this Elixirforum post gives some directions: <https://elixirforum.com/t/rag-app-using-elixir-feasible/60439/15>

