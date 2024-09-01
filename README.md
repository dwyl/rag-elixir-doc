# rag-elixir-doc

<h1 align="center">Building LLM enhanced search with the help of LLMs....</h1>

We want to improve the search for the Elixir/Phoenix/Plug/LiveView documentation when using an LLM and experiment a `RAG` pipeline.

All the tools used here are "free", meaning everything is running locally.

[![Run in Livebook](https://livebook.dev/badge/v1/blue.svg)](https://livebook.dev/run?url=https%3A%2F%2Fgithub.com%2Fdwyl%2Frag-elixir-doc%2Fblob%2Fmain%2Frag-elixir.livemd)

## What is `RAG`?

It is a "chat with your documents" process, meaning you ask an LLM model to respond based on additional ressources.

Theses sources may or may not be already incorporated inside the training used for the LLM.

Using RAG is _not about fine tuning_ the model, which is changing the coefficients or structure of the model based on additional sources.

RAG is about giving an additional context - the "context window" - to enhance or constraint the response from the LLM.

> Note that the LLM accepts a limited amounts of tokens, thus the window context is limited.


## Scope of this POC:

We want to improve the LLm's response when we ask questions related to the Elixir/Phoenix/Plug/LiveView documentation. We will build a "context" to add information to help the LLM to build a response.

Running such a helper locally means that we need to have the extra ressources available locally. Our database will be local and our LLM will be run locally, using only local ressources.

We will extract some markdown files from the Phoenix_LiveView GitHub repo.

We will use a database to store chunks extracted from these files.


- One way is **SQL Full-Text-Search**. If we use `Postgres`, we have a [built-in functionality](https://www.postgresql.org/docs/current/textsearch-intro.html#TEXTSEARCH-DOCUMENT). This works by using ** key words**, so given that we may have various ways to express the same questions, we may want a more semantic search. [A good introduction to SQL FTS](https://leandronsp.com/a-powerful-full-text-search-in-postgresql-in-less-than-20-lines) or [here](https://peterullrich.com/complete-guide-to-full-text-search-with-postgres-and-ecto).

- We can also use the open source version of "elastisearch" by Amazon named `opensearch` via a Docker image. We may also use the [local image of ElastiSearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html). There exists [Text Chunking tools](https://opensearch.org/docs/latest/search-plugins/text-chunking/).  The matching is based on the **BM25** algorithm. This might be an interesting extension to use this instead of the **bi-encoding** step, as a second version. Indeed, Elixir has an Opensearch client: [snap](https://github.com/breakroom/snap)

- we can build a RAG pipeline with an LLM. We follow this route here. We save chunks with their embedding into a database, run a similarity search against our query, and then rerank our top-k reponses, and finally use the last reponse to build our prompt and submit it as a context to the LLM. This will help to constrain responses on questions about Elixir/Plug/Phoenix_Liveview. 


## Using LLM to find which models to use

We are using pre-trained NR models. We will also use an LLM to elaborate a response. 

The first question is: what does a RAG pipeline brings?

We believe the next two paragraphs give some confidence that this technic works, or at least improves some responses.

### Testing Codellama

```text
Question: "how to handle forms server-side?"
```

<details><summary>The "raw" response</summary>

Handling forms on the server-side involves using the `Phoenix.HTML` module's `form_for` function to create a form struct with the necessary information for submitting the form data to the server.
Here's an example:

```elixir
import Phoenix.HTML

# Create a form struct with the necessary information
form = form_for(conn, :action, :controller)

# Add fields to the form using the `add_field` function
form = form |> add_field(:name, :text) |> add_field(:age, :number)

# Use the `handle_event` function to handle form submissions
handle_event(form, :submit, conn) do
  # Do something with the submitted data here, e.g. update a database
end
```

The `form_for` function takes three arguments:

	* `conn`: The current connection struct
	* `action`: The action to perform when submitting the form(e.g. `:create`, `:update`, etc.)
	* `controller`: The controller that handles the form submission

Once you have created the form struct, you can add fields to it using the `add_field` function.
This function takes two arguments:

	* `name`: The name of the field (e.g. `:username`,`:password`, etc.)
	* `type`: The type of field (e.g. `:text`, `:number`, etc.)

You can then use the `handle_event` function to handle form submissions. This function takes three arguments:

	* `form`: The form struct that was created using the `form_for` function
	* `event`: The event that triggered the submission (e.g. `:submit`, etc.)
	* `conn`: The current connection struct

Inside the `handle_event` function, you can perform any necessary actions when the form is submitted, such as updating a database or sending an email.

To handle form validation and error handling, you can use the `Phoenix.LiveView` module's `validate` and `error` functions.

For example:

```elixir
import Phoenix.LiveView

# Add validation rules to the form using the `validate` function
form = form |> validate(:name, :required) |> validate(:age, :number)

# Handle errors in the form submission using the `error` function
handle_event(form, :submit, conn) do
  if error?(conn) do
    # Do something with the validation errors here, e.g. display an error message
  else
    # Do something with the submitted data here, e.g. update a database
  end
end
```

In this example, we use the `validate` function to add validation rules to the form. The `:required` and `:number` arguments specify that the fields should be required and contain only numbers, respectively.
If there are any errors in the submitted data, the `error?` function will return true and we can handle them using the `error` function.
</details>
<br/>

<details><summary>The "RAGed" response</summary>

To handle forms with LiveView, you can use the `Phoenix.Component` module and its functions to define a form component that will submit and validate the form data using LiveView's built-in validation mechanism. Here's an example of how to define a form component in your LiveView:

```elixir
defmodule MyAppWeb.Components.Form do
  use Phoenix.Component
  def form(assigns) do
    ~H"""
      <.form for={@form} phx-change="validate" phx-submit="save">\
        ...
      </.form>
    """
    end
  end
...
```

In the above example, we define a `Form` component that renders a form tag with a `phx-change` and `phx-submit` event to handle validation and submission of the form data. The `for` attribute is used to bind the form to a changeset or user parameters, which can be created from a form struct using `Phoenix.Component.to_form/1`.

You may also take advantage of LiveView's CSS loading state classes to swap out your form content while the form is submitting. For example, with the following rules in your `app.css`:

```css
.while-submitting { display: none; }
.inputs { display: block; }

.phx-submit-loading .while-submitting { display: block; }
  
.phx-submit-loading .inputs { display: none; }
```

You can show and hide content with the following markup:

```elixir
<.form for={@form} phx-change="validate" phx-submit="save">
  <div class="while-submitting">Please wait while we save our content...</div>
  <div class="inputs">
    <input type="text" name="text" value={@text}>
</div>
</.form>
```

Additionally, we strongly recommend including a unique HTML "id" attribute on the form. When DOM siblings change, elements without an ID will be replaced rather than moved, which can cause issues such as form fields losing focus.

You may wish for an individual input to use its own change event or to target a different component. This can be accomplished by annotating the input itself with `phx-change`, for example:

```elixir
<.form for={@form} phx-change="validate" phx-submit="save">
  ...
  <.input field={@form[:email]}  phx-change="email_changed" phx-target={@myself} />
</.form>
```

The `phx-change` event will trigger the `email_changed` function on the `@myself` component, which can be used to update the form data or perform other actions based on changes to the input field.

You may also wish for an individual input to use its own change event or to target a different component. This can be accomplished by annotating the input itself with `phx-change`, for example:

```elixir
<.form for={@form} phx-change="validate" phx-submit="save">
  ...
  <.input field={@form[:email]}  phx-change="email_changed" phx-target={@myself} />
</.form>
```

The `phx-change` event will trigger the `email_changed` function on the `@myself` component, which can be used to update the form data or perform other actions based on changes to the input field.\n\nIt's also important to note that you may want to use a unique HTML \"id\" attribute on the form. When DOM siblings change, elements without an ID will be replaced rather than moved, which can cause issues such as form fields losing focus.\n\nYou may also wish for an individual input to use its own change event or to target a different component. This can be accomplished by annotating the input itself with `phx-change`, for example:

```elixir
<.form for={@form} phx-change="validate" phx-submit="save">
  ...
  <.input field={@form[:email]}  phx-change="email_changed" phx-target={@myself} />
</.form>
```

The `phx-change` event will trigger the `email_changed` function on the `@myself` component, which can be used to update the form data or perform other actions based on changes to the input field.
</details>
<br/>

The first conclusion is that the RAG pipeline improves a lot the response.
This happens also because most probablythe current version of the LLM (July 2024) is not up-to-date with the current state-of-art at the time of writting.

We continue by asking the LLM to help us with the technical task below. We encounter the main limitation: halucination.

<details><summary>Which embedding models can we use? Some examples of how LLMs respond to this question</summary>

We know that most questions that are submitted to an LLM will produce so-called "halucinated" responses, or in other words, invented.

Let's test this with LLama3.1, choosen because you can run it locally via `ollama run llama3.1`.

```txt
question: "I want to use an Elixir portage of llama. Can you suggest a library?

response: "**Elixir-T5**: This is an Elixir port of a popular LLM architecture (T5). It's designed specifically for use with Elixir and might be worth considering. You may also consider **LLaMA-erl**: This is an Elixir wrapper for the LLaMA model, which is similar to Ollama.
```

Nor `Elixir-T5` nor `LLama-erl` exist!

I tried CodeLLama: `ollama run codellama`:

```txt
question: which Elixir library would you suggest for text chunking to compute embeddings efficiently?
```

<details><summary>codellama response</summary>

```console
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
</details>
<br/>

We check that `elixir_bert` does not exist, nor `elixir_bert_embedding`, nor `elixir_glove` nor `elixir_workd2vec`. 

However, after researching a bit, the library `ExFinalFusion` exists, a NIF of `FinalFusion`.
[![Screenshot 2024-08-24 at 19 33 08](https://github.com/user-attachments/assets/11d7daeb-8f3f-49ea-a2af-38c2c654c42d)](https://hexdocs.pm/ex_final_fusion/ExFinalFusion.html#summary)

### Let the LLMs choose the models

The question:

```console
I want to code a RAG pipeline in the Elixir langugage using its ML libraries (Nx, Axon, Bumble
bee). I want to use a bi-encoder to compute embeddings from chunks of text, and then run a
cross-encoding process. Can you recommend  models firstly for the bi-encoding and then for the
cross-encoding step, having in mind that I need a model for tokenizing the input.
```

Some reponses with differents LLMs.

<details><summary>codellama response</summary>

```console
Bi-encoding (Tokenization):

* BERT (Bidirectional Encoder Representations from Transformers)
	+ This is a popular pre-trained language model that can be fine-tuned for various NLP tasks,
including tokenization.
	+ BERT uses a multi-layer bidirectional transformer encoder to generate contextualized
representations of words in a sentence.
* RoBERTa (Robustly Optimized BERT Pretraining Approach)
	+ This is a variant of BERT that was specifically designed for text classification tasks and
has been shown to achieve state-of-the-art results on several benchmark datasets.
	+ RoBERTa uses a combination of tokenization and language modeling techniques to generate

contextualized representations of words in a sentence.
* DistilBERT (Distilled BERT)
	+ This is a smaller and more efficient variant of BERT that has been trained to match the

performance of the full BERT model while requiring fewer computational resources.
	+ DistilBERT uses a combination of tokenization and attention techniques to generate
contextualized representations of words in a sentence.

Cross-encoding:

* MLP (Multi-layer perceptron)
	+ This is a simple feedforward neural network that can be used for classification tasks such
as text classification, sentiment analysis, and topic modeling.
	+ An MLP consists of multiple layers of artificial neurons that learn to represent the input
data in a higher dimensional space.
* CNN (Convolutional neural network)
	+ This is a deep learning architecture that is particularly well-suited for text
classification tasks where the input data has a spatial or temporal structure.
	+ A CNN consists of multiple convolutional layers that learn to extract features from the

input data, followed by pooling and fully connected layers to make predictions.

It's important to note that the specific models you choose will depend on your specific use
case and requirements. You may also want to consider using a combination of different models
to achieve better results.
```
</details>
<br/>

<details><summary>Llama3 response</summary>

```console
**Bi-Encoding Step: Tokenization and Embedding**

For tokenization, you can use a simple word tokenizer like `Nx.Tokenizer` from the Elixir
`axon` library. However, if you need more advanced tokenization capabilities (e.g., handling
out-of-vocabulary words), consider using Bumblebee's `Bee.Tokenizer`. For embedding the
tokens into vectors, I recommend using:

1. **Universal Sentence Encoder (USE)**: This is a pre-trained language model that can be
used for token embedding. You can use the Elixir wrapper provided by Axon to load and use
the USE model.

**Cross-Encoding Step**

For cross-encoding, you'll need to take the output of the bi-encoding step and generate a
new representation based on it. I recommend using:

1. **Reformer**: This is a state-of-the-art transformer architecture that's well-suited for
the RAG pipeline. You can use Bumblebee to load and use the Reformer model.

Here are some additional recommendations:

* For tokenizing input text, you might want to consider using more advanced techniques like
subwording (e.g., WordPiece) or character-level encoding.
* If you're dealing with long documents, you may need to split them into smaller chunks
before applying bi-encoding and cross-encoding. Consider using a technique like document
chunking or summarization.

To give you a better understanding of the pipeline flow, here's a high-level outline:

1. **Tokenize input text**: Use `Nx.Tokenizer` or Bumblebee's `Bee.Tokenizer` to break down
the text into individual tokens.
2. **Bi-encode tokens**: Apply USE to embed each token into a vector space (e.g.,
128-dimensional).
3. **Cross-encode output**: Use Reformer to generate a new representation based on the
bi-encoded output.

Remember to preprocess your data (e.g., normalize text, remove stop words) before feeding it
into these models.

For more information and implementation details, I recommend checking out the following
resources:

* [Bumblebee documentation](https://hexdocs.pm/bumblebee/)
* [Axon documentation](https://hexdocs.pm/axon/)
* [Elixir ML
libraries](https://elixir-lang.org/docs/master/libraries.html#machine-learning-libraries)
* [RAG paper and implementation](https://arxiv.org/abs/2006.16262)
```
</details>
<br/>

<details><summary>Claude3.5 Sonnet response</summary>

```console
Bi-encoding step:

For the bi-encoding step, where you'll encode both the chunks from the documentation and the query, a good choice would be a sentence transformer model. A popular option that works well for semantic search is:

Model: "sentence-transformers/all-MiniLM-L6-v2"

This model provides a good balance between performance and speed. It's relatively small (80MB) but still gives good results for semantic similarity tasks.

Cross-encoding step:

For the cross-encoding step, where you'll want to get a more precise relevance score between the query and the retrieved documents, you could use:

Model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

This model is specifically trained for relevance ranking and should work well for your cross-encoding needs.

Tokenization:

For tokenization, you'll want to use a tokenizer that matches the models you're using. Since both recommended models are based on MiniLM, you can use the MiniLM tokenizer:

Tokenizer: "microsoft/MiniLM-L6-H384-uncased"
```
</details>
<br/>

<details><summary>ChatGPT response</summary>

```console
To create a Retrieval-Augmented Generation (RAG) pipeline in Elixir using the Nx, Axon, and Bumblebee libraries, you'll need to carefully choose models that are suitable for both the bi-encoding and cross-encoding processes. Let's break down the components:

1. Bi-Encoder Step:
Purpose: The bi-encoder step involves encoding chunks of text independently to generate embeddings that can be used for retrieval.
Recommended Models:
MiniLM (Microsoft): A small, efficient model designed for sentence embeddings. It's often used for bi-encoder tasks due to its balance between speed and accuracy.
DistilBERT: A smaller and faster version of BERT, ideal for generating embeddings from text while maintaining a good balance of performance.
Sentence-BERT (SBERT): This is a modification of BERT that is specifically designed for creating dense vector representations of sentences, making it a solid choice for bi-encoding.
2. Cross-Encoder Step:
Purpose: The cross-encoder step involves comparing the query and retrieved passages together, typically used to re-rank the retrieved results.
Recommended Models:
BERT: The standard BERT model is highly effective for cross-encoding tasks because it considers the interaction between the query and the passage.
RoBERTa: A variant of BERT with more robust training, which tends to perform better in NLP tasks.
DeBERTa: Known for its improved handling of word order and dependency information, making it a strong candidate for cross-encoding.
3. Tokenization:
Purpose: Tokenizing the input is crucial for both the bi-encoder and cross-encoder steps. The tokenizer needs to be compatible with the chosen models.
Recommended Tokenizers:
BERT Tokenizer: Compatible with BERT-based models (BERT, RoBERTa, DistilBERT, etc.).
SentencePiece Tokenizer: Often used with models like MiniLM and other transformer-based models.
Elixir Implementation:
Libraries:
Nx: For numerical computation, handling tensors, and integrating with the underlying ML operations.
Axon: For building neural networks, including the bi-encoder and cross-encoder models.
Bumblebee: For pre-trained models, including tokenizers and transformers, which will help with the embedding and cross-encoding tasks.
Example Models in Bumblebee:
For bi-encoding: Use models like "MiniLM-L6-v2" or "distilbert-base-uncased" from Bumblebee.
For cross-encoding: Use models like "bert-base-uncased" or "roberta-base".
Steps to Implement:
Load the Model: Use Bumblebee to load the bi-encoder and cross-encoder models.
Tokenize the Input: Tokenize the text using the appropriate tokenizer.
Compute Embeddings: Generate embeddings using the bi-encoder model.
Retrieve and Rank: Use the cross-encoder model to compare and rank the retrieved documents.
```
</details>


</details>

### First conclusion for the LLM

It seems that Claude3.5 Sonnet and ChatGTP give much "better" responses than Llama. However, we need an API to reach their REST API whilst we can run la local version of the LLama LLM. 

We will therefore choose LLama, running it at no cost, knowing that it can only get better by choosing a "paid" version with OpenAI or better Anthropic.

### Conclusion for the choice of our models

We will use SBERT based models:
- the bi-encoder"sentence-transformers/all-MiniLM-l6-v2" which also provides a tokenizer,
- the cross-encoder "cross-encoder/ms-marco-MiniLM-L-6-v2" along with the tokenizer "bert-base-uncased"


## Source of knowledge

We firstly seed the vector database with some Github markdown pages from the Elixir documentation.

The sources will be extracted from the files that the GitHub API returns when querying some directories:
  - <https://api.github.com/repos/phoenixframework/phoenix_live_view/contents/guides/server/>
  - <https://api.github.com/repos/phoenixframework/phoenix_live_view/contents/guides/client/>
  - <https://github.com/phoenixframework/phoenix_live_view/blob/main/guides/introduction/welcome.md>

  - we can also add some ".ex" modules when they provide documentation in a moduledoc.


## Overview of the RAG process:
  * installed tools: the database `Postgres` with the `pgvector` extension, the plateform `ollama` to run LLM locally.
    
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

The pipeline will use three SBert based models: "sentence-transformers/all-MiniLM-L6-v2" for the embedding, "cross-encoder/ms-marco-MiniLM-L-6-v2" for the reranking, and "bert-base-uncased" for tokenizing.

In pseudo-code, we have:
  
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

- [SBert](https://www.sbert.net/): check: <https://huggingface.co/spaces/mteb/leaderboard>.


## Vector database of Index?

- An index [HNSW](https://github.com/elixir-nx/hnswlib), the Elixir portage of `hnswlib`, a KNN search,
- or a vector database?
	- Postgres with [pgvector](https://github.com/pgvector/pgvector) with the Elixir portage: [pgvector-elixir](https://github.com/pgvector/pgvector-elixir),
 	- SQLite with [sqlite-vec](https://github.com/asg017/sqlite-vec). The extension has to be installed manually from the repo and loaded (with `exqlite`),
	- or [Supabase](https://github.com/supabase/supabase), with an [Elixir client](https://github.com/zoedsoupe/supabase-ex)
	- or [ChromaDB](https://github.com/3zcurdia/chroma), with an [Elixir client](https://github.com/3zcurdia/chroma)

We will use Postgres with the extension `pgvector` and the `HNSW` algorithm. See discussion on the Postgres + pg_vector setup.
  
## How to **prompt**? 

This is where we define the scope of the response we want from the LLM, given the retrieved context given by the database nearest neighbour search. 

The LLM should be able to generate an "accurate" response constrainted by this context.

## A word on **LLMs**

A Dockyard post on this: <https://dockyard.com/blog/2023/05/16/open-source-elixir-alternatives-to-chatgpt>.

A comparison of different LLMs (source: Anthropic)
<img width="592" alt="Screenshot 2024-08-14 at 17 56 40" src="https://github.com/user-attachments/assets/af4ef9ea-88f8-42bf-b963-013ea35d429f">

### Pricing

[![Screenshot 2024-08-28 at 21 19 41](https://github.com/user-attachments/assets/a66e8689-a7c2-46e6-a597-fb141426b9cf)](https://openai.com/api/pricing/)

[![Screenshot 2024-08-28 at 21 21 51](https://github.com/user-attachments/assets/081a4b1c-579d-4801-b441-d7f14be9c76a)](https://www.anthropic.com/pricing#anthropic-api)

## Going further?

- accept new documents "on the fly" (download a given link), and maybe running the database ingesting in a background job.


- use `Opensearch` instead of the bi-encoding:
	- install local:
 		- <https://opensearch.org/docs/latest/install-and-configure/install-opensearch/docker/>
 		- <https://waytohksharma.medium.com/opensearch-local-running-instance-with-docker-with-m1-1e60d90a263c>
   	- ingest data: <https://opensearch.org/docs/latest/getting-started/ingest-data/> and <https://opensearch.org/docs/latest/api-reference/document-apis/index-document/>

- clusterise data?

## Source of inspiration. 

Which repos, blog post?
  - <https://dockyard.com/blog/2024/05/16/retrieval-augmented-generation-what-it-is-how-to-use-it>
  - using the cross-encoder: <https://github.com/elixir-nx/bumblebee/issues/251>
  - Bumblebee, RAG: <https://hexdocs.pm/bumblebee/llms_rag.html#introduction>
  - Supabase: <https://github.com/supabase-community/chatgpt-your-files>
  - Langchain: <https://github.com/brainlid/langchain_demo>
  - <https://dockyard.com/blog/2024/05/16/retrieval-augmented-generation-what-it-is-how-to-use-it>
  - <https://github.com/nileshtrivedi/autogen>
  - <https://dockyard.com/blog/2023/05/16/open-source-elixir-alternatives-to-chatgpt>
  - <https://fly.io/phoenix-files/using-llama-cpp-with-elixir-and-rustler/>
  -  A Fly.io post on using ``llama.cpp` with `Rustler`: <https://fly.io/phoenix-files/using-llama-cpp-with-elixir-and-rustler/> 
  -  ExLLama: LlammaCpp.rs NIF wrapper for Elixir/Erlang: <https://hexdocs.pm/ex_llama/readme.html> and <https://fly.io/phoenix-files/using-llama-cpp-with-elixir-and-rustler/>
  -  ollama-ex to run LLM locally: <https://hexdocs.pm/ollama/Ollama.html>
    

