# rag-elixir-doc

<h1 align="center">Building....</h1>

## What is `RAG`?

It is "chat with your documents", meaning you ask an LLM model to respond based on additional sources.

Theses sources may or may not be already incorporated inside the training used for the LLM.

Using RAG is _not about fine tuning_ the model, which is changing the coefficients or structure of the model based on additional sources.

RAG is about giving an additional context - the "context window" - to enhance or constraint the response from the LLM.

> Note that the LLM accepts a limited amounts of tokens, thus the window context is limited.


## Scope of this POC:


We want to improve the search for the Elixir/Phoenix/Plug/LiveView documentation.

One way could be to have a RAG pipeline with an LLM. This is a POC of a RAG where we inject the documentation and see if this can help to find "precise" responses or help to constrain responses on questions about Elixir/Plug/Phoenix_Liveview. 


#### Halucination

We know that most questions submitted to an LLM will produce halucinated responses, or in other words, invented.
Let's test this with LLama3.1, choosen because you can run it locally via `ollama run llama3.1`.

```txt
question: "I want to use an Elixir portage of llama. Can you suggest a library?

response: "**Elixir-T5**: This is an Elixir port of a popular LLM architecture (T5). It's designed specifically for use with Elixir and might be worth considering. You may also consider **LLaMA-erl**: This is an Elixir wrapper for the LLaMA model, which is similar to Ollama.
```

Nor `Elixir-T5` nor `LLama-erl` exist!

I tried CodeLLama: `ollama run codellama`:

```txt
question: which Elixir library would you suggest for text chunking to compute embeddings efficiently?

response:
There are several Elixir libraries available for text chunking and computing embeddings, each with its own strengths and weaknesses. Here are a few popular options:

1. `elixir_bert`: This is an Elixir port of the BERT (Bidirectional Encoder Representations from Transformers) model, which is a powerful language model that can be
used for text chunking. It provides a simple and efficient interface for computing embeddings, and it supports both CPU and GPU acceleration.
2. `elixir_bert_embedding`: This is an Elixir wrapper for the BERT embedding layer, which allows you to compute embeddings for input sequences using a pre-trained
BERT model. It provides a simple interface for text chunking and supports both CPU and GPU acceleration.
3. `elixir_word2vec`: This is an Elixir port of the Word2Vec model, which is another popular language model that can be used for text chunking. It provides a simple
and efficient interface for computing embeddings, and it supports both CPU and GPU acceleration.
4. `elixir_glove`: This is an Elixir port of the GLOVE (Global Vectors for Word Representation) model, which is a language model that can be used for text chunking.
It provides a simple and efficient interface for computing embeddings, and it supports both CPU and GPU acceleration.

All of these libraries should work well for your purposes, but you may want to consider which one best fits your specific needs. If you have any specific requirements
or preferences (e.g., support for a particular file format or feature), you may want to choose the library that is most suitable for your use case.
```

`elixir_bert` does not exist, nor `elixir_bert_embedding`, nor `elixir_glove` nor `elixir_workd2vec`. However, after researching a bit, I found `ExFinalFusion`, a NIF of `FinalFusion`.
[![Screenshot 2024-08-24 at 19 33 08](https://github.com/user-attachments/assets/11d7daeb-8f3f-49ea-a2af-38c2c654c42d)](https://hexdocs.pm/ex_final_fusion/ExFinalFusion.html#summary)


## Source of knowledge

We firstly seed the vector database with some Github markdown pages from the Elixir documentation.

The sources will be extracted from the files that the GitHub API returns when querying some directories:
  - <https://api.github.com/repos/phoenixframework/phoenix_live_view/contents/guides/server/>
  - <https://api.github.com/repos/phoenixframework/phoenix_live_view/contents/guides/client/>
  - <https://github.com/phoenixframework/phoenix_live_view/blob/main/guides/introduction/welcome.md>

  - we can also add some ".ex" modules when they provide documentation in a moduledoc.


## Overview of the process:
  
  * Build the external sources.
    - Download "external sources" as a string
    - chunk the sources
    - produce an embedding based on a "sentence-transformer" model for each chunk
    - insert chunk + embedding into a Vector database using a HSNW index
      
  * Build a RAG pipeline
    - produce an embedding (a vector representation) from the question
    - perform a first vector similarity search (HNSW) against the database
    - rerank the top-k with "cross-encoding"
    - build a prompt by injecting the later result with the query as a context
    - submit the prompt to the LLM for completion

### Pseudo-code pipeline

The pipeline will use two models
  
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

## What is **bi-encoding** and **cross-encoding**?

- [Bi-encoders]: Encode the query and document separately, then compare their vector representations. This is the "standard" similarity search.

  Bi-encoding does consider the relationship between the query and each document, but it does so independently for each document. The main problem is that bi-encoding might not capture nuanced differences between documents or complex query-document relationships. `HNSW` indexes or `BM25` can be used for this.
  
- Cross-encoders: Take both the query and document as input simultaneously, allowing for more complex interactions between them. It processes the query and document together through a neural network (typically a transformer model like BERT) to produce a single relevance score. This allows the model to capture complex interactions between the query and document at all levels of representation.

  Cross-encoders typically perform better than bi-encoders in terms of accuracy, but are computationally more expensive and slower at inference time.
  They are not suitable for large-scale retrieval because they require comparing the query with every document from scratch, which doesn't scale well.
  Therefor, Cross-encoding is often used in a two-stage retrieval process.

- How cross-encoders works in reranking?:
  - After initial retrieval (e.g., using vector similarity), you pass each query-document pair through the cross-encoder.
  - The cross-encoder outputs a relevance score for each pair.
  - Results are then sorted based on these scores, potentially significantly changing the order from the initial retrieval.
    
  
## How to **chunk**? 

We need to define how to ingest these documents to produce _embeddings_ saved into a _vector database_. 

Do we run a naive chunk? or [use this package](https://github.com/revelrylabs/text_chunker_ex), or [structured chunks](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_vs_recursive_retriever/), [Chunk + Document Hybrid Retrieval](https://docs.llamaindex.ai/en/stable/examples/retrievers/multi_doc_together_hybrid/), or use [BM25](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/), with an Elixir implementation [BM25](https://github.com/elliotekj/bm25)? 

## Which embedding? 

- [SBert](https://www.sbert.net/),
- or as in [this video](https://www.youtube.com/watch?v=ibzlEQmgPPY) uses "GT-SMALL" (from Alibaba)
- or "sentence-transformer". Check: <https://huggingface.co/spaces/mteb/leaderboard>.

We opted for "sentence-transformer".

## Vector database of Index?

- An index (`HNSW`)
- or a vector database? Postgres with PGVector,
- or [Supabase](https://github.com/supabase/supabase),
- or [ChromaDB](https://github.com/3zcurdia/chroma)?

We will use Postgres with the extension PG_Vector and the HNSWL algorithm. See discussion on the Postgres + pg_vector setup.
  
## How to **prompt**? 

This is where we define the scope of the response we want from the LLM, given the retrieved context given by the database nearest neighbour search. 

The LLM should be able to generate an "accurate" response constrainted by this context.

## Which **LLM**? 

Our main problem will be halucination, in other words, the LLM invents the response.

The LLM could be OpenAI ChatGPT 3.5/4? Meta Llama? Mistral? Anthropic Claude 3.5 Sonnet? , Google T5?

A Dockyard post on this: <https://dockyard.com/blog/2023/05/16/open-source-elixir-alternatives-to-chatgpt>.

#### LLama ?

- A Fly.io post on using ``llama.cpp` with `Rustler`: <https://fly.io/phoenix-files/using-llama-cpp-with-elixir-and-rustler/>

[![Screenshot 2024-08-24 at 18 58 16](https://github.com/user-attachments/assets/9bdc74d8-4e83-4ffc-8ac1-70a941cdf55b)](https://fly.io/phoenix-files/using-llama-cpp-with-elixir-and-rustler/)

- the "ollama-ex" wrapper:
  [<img width="974" alt="Screenshot 2024-08-23 at 17 21 44" src="https://github.com/user-attachments/assets/cd86b1c3-9bdb-46cb-a4df-20cf34604380">](https://hexdocs.pm/ollama/Ollama.html)

- ExLLama: LlammaCpp.rs NIF wrapper for Elixir/Erlang:
[![Screenshot 2024-08-24 at 18 58 28](https://github.com/user-attachments/assets/8b7194ce-4ecf-4c84-87d7-395d3100a276)](https://hexdocs.pm/ex_llama/readme.html)

- a comparison of different LLMs:
<img width="592" alt="Screenshot 2024-08-14 at 17 56 40" src="https://github.com/user-attachments/assets/af4ef9ea-88f8-42bf-b963-013ea35d429f">

### Pricing

[![Screenshot 2024-08-28 at 21 19 41](https://github.com/user-attachments/assets/a66e8689-a7c2-46e6-a597-fb141426b9cf)](https://openai.com/api/pricing/)

[![Screenshot 2024-08-28 at 21 21 51](https://github.com/user-attachments/assets/081a4b1c-579d-4801-b441-d7f14be9c76a)](https://www.anthropic.com/pricing#anthropic-api)

## Going further?

This POC is expected to be run **only locally** (no deploy, mainly because of the costs).

Can we measure our performance?

We can enhance our documentation base by accepting documents "on the fly" (download a given link).


## Source of inspiration. 

Which repos, blog post?
  - <https://dockyard.com/blog/2024/05/16/retrieval-augmented-generation-what-it-is-how-to-use-it>
  - Bumblebee, RAG: <https://hexdocs.pm/bumblebee/llms_rag.html#introduction>
  - Supabase: <https://github.com/supabase-community/chatgpt-your-files>
  - Langchain: <https://github.com/brainlid/langchain_demo>
  - <https://dockyard.com/blog/2024/05/16/retrieval-augmented-generation-what-it-is-how-to-use-it>
  - <https://github.com/nileshtrivedi/autogen>
  - <https://dockyard.com/blog/2023/05/16/open-source-elixir-alternatives-to-chatgpt>
  - <https://fly.io/phoenix-files/using-llama-cpp-with-elixir-and-rustler/>

