# rag-elixir-doc

Create a POC of a RAG to help/improve searching in the Elixir/Phoenix/Plug/LiveView documentation.

- which LLM? The base LLM could be ChatGPT 3.5? Ollama? Claude 3.5 Sonnet: performance vs costs 😥

<img width="592" alt="Screenshot 2024-08-14 at 17 56 40" src="https://github.com/user-attachments/assets/af4ef9ea-88f8-42bf-b963-013ea35d429f">

- Which embedding? To be defined. [This video](https://www.youtube.com/watch?v=ibzlEQmgPPY) uses "GT-SMALL" (from Alibaba).
  The problem could be to be able to use this with Elixir/Nx/Axon/Bumblebee. The list: <https://huggingface.co/spaces/mteb/leaderboard>

- What is our source? We can firstly only seed the vector database with some Github raw pages we download.
  We need to define how to parse the documents and how to produce these embeddings.

- Which database? Postgres with PGVector, or Supabase??
  
- Which interface? A very simple one: an input that takes a text, and a textarea where we display the response.

- Next? We can further improve be accepting documents, possibly links serving raw text, but also upload markdown files.

- Deploy this? As a first step, this is a POC, to be run locally.

- Source of inspiration. Repos, blog post?
  - <https://github.com/supabase-community/chatgpt-your-files>
  - <https://github.com/nileshtrivedi/autogen>

What are your thoughts?
