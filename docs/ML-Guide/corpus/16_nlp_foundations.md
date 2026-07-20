# NLP Foundations for ML Interviews

The job description says "Familiarity with Natural Language Processing (NLP) techniques and models is desired." It is preferred, not required, but you should still know enough to discuss and implement basic NLP pipelines.

## Why NLP Matters in Production

Most business text data is messy and unstructured:
- Customer support tickets
- Product reviews
- Survey responses
- Search queries
- Chat messages
- Documents and emails
- Social media posts

Turning this text into structured predictions (intent, sentiment, topic, relevance) is a common business need.

## The NLP Pipeline

A typical NLP pipeline:

1. **Text acquisition**: collect raw text from logs, APIs, databases.
2. **Cleaning**: remove HTML, normalize unicode, handle encoding.
3. **Tokenization**: split text into tokens (words, subwords, characters).
4. **Normalization**: lowercase, lemmatize, stem, remove stopwords (optional).
5. **Feature extraction**: TF-IDF, word embeddings, contextual embeddings.
6. **Modeling**: classification, regression, generation, extraction.
7. **Evaluation**: task-specific metrics.
8. **Deployment**: latency budget, model size, monitoring.

## Tokenization

Tokenization splits text into units (tokens). Choice of tokenizer matters a lot.

### Word-level tokenization

Split on whitespace and punctuation. Simple but:
- Large vocabulary (one entry per word).
- Cannot handle out-of-vocabulary words.
- Does not generalize across languages.

### Subword tokenization (BPE, WordPiece, SentencePiece, Unigram)

Splits rare words into common subwords. Modern standard.

Examples:
- "unhappiness" → ["un", "happiness"] or ["un", "happy", "ness"]
- "chatgpt" → ["chat", "gp", "t"]

**Byte-Pair Encoding (BPE)**: iteratively merges the most frequent pair of adjacent tokens. Used by GPT.

**WordPiece**: similar to BPE but uses likelihood-based merges. Used by BERT.

**SentencePiece**: language-agnostic; treats text as a stream of Unicode chars. Used by many multilingual models.

**Unigram**: starts with a large vocabulary and prunes. Used by some models.

Most modern models (BERT, RoBERTa, GPT, T5) come with their own tokenizers. Use them; do not roll your own.

### Character-level tokenization

Each character is a token. Very small vocabulary, handles any input, but sequences are long and slower to train.

## Classical NLP Features

Even in 2026, classical features are still useful as baselines.

### Bag of Words (BoW)

Count the occurrences of each word in a document. Each document is a vector of word counts.

Pros: simple, fast, interpretable.
Cons: ignores word order, high dimensional, no semantics.

### TF-IDF (Term Frequency - Inverse Document Frequency)

Weights words by their importance: frequent in the document, rare in the corpus.

```
TF(t, d) = count(t in d) / total_words(d)
IDF(t) = log(N / df(t))
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

- Common words ("the", "is") get low TF-IDF (high DF).
- Rare but meaningful words get high TF-IDF.
- Standard feature for classical text classification.

### n-grams

Sequences of n consecutive words. "I love this" has unigrams {I, love, this}, bigrams {I love, love this}.

Captures local word order. Useful for sentiment ("not good" vs "good").

### Word embeddings (Word2Vec, GloVe, FastText)

Pre-trained dense vector representations of words.

- **Word2Vec**: predicts context from word (or vice versa). Two flavors: CBOW and Skip-gram.
- **GloVe**: factorization of word co-occurrence matrix.
- **FastText**: subword embeddings; handles out-of-vocabulary by summing subword vectors.

Properties:
- Words with similar meaning have similar vectors.
- Vector arithmetic: king - man + woman ≈ queen.
- 50-300 dimensions, much denser than BoW/TF-IDF.

Limitations:
- One vector per word (no context).
- Cannot distinguish "bank" (river) from "bank" (money).

### Contextual embeddings (ELMo, BERT, etc.)

The current standard. Word vectors depend on context.

- **ELMo**: bidirectional LSTM; embeddings vary by context.
- **BERT**: bidirectional transformer; pre-trained on MLM + NSP.
- **RoBERTa, DeBERTa, ELECTRA**: BERT improvements.
- **Sentence transformers**: BERT fine-tuned for sentence-level similarity.

The same word in different contexts gets different vectors. "Bank of the river" and "bank account" are now distinguishable.

## Common NLP Tasks

### Text classification

Assign a label to a text (sentiment, topic, intent, toxicity, etc.).

Approaches:
1. TF-IDF + logistic regression / linear SVM (strong baseline).
2. Fine-tuned BERT (when you need top performance and have enough data).
3. LLM-based zero-shot or few-shot (when you have no training data).

Metrics: accuracy, F1, PR-AUC (for imbalanced classes).

### Named Entity Recognition (NER)

Identify and classify entities in text (people, organizations, locations, dates).

Approaches:
- Conditional random fields (CRF) + hand-crafted features.
- BiLSTM + CRF.
- Fine-tuned BERT with token classification head.

### Sentiment analysis

Determine polarity (positive/negative/neutral) or intensity.

Often framed as classification. Can also be regression (predict sentiment score).

### Text similarity and search

Find similar documents or sentences.

Approaches:
- **BM25**: classical ranking function based on TF-IDF. Strong baseline.
- **Dense retrieval**: encode text into vectors, search by cosine similarity.
- **Hybrid**: combine BM25 and dense retrieval, rerank with a cross-encoder.

### Question answering and reading comprehension

Given a question and context, extract or generate the answer.

Approaches:
- **Extractive QA**: BERT with span prediction head (start, end positions).
- **Generative QA**: encoder-decoder (T5) or decoder-only (GPT) models.
- **RAG (Retrieval-Augmented Generation)**: retrieve relevant context, then generate.

### Text generation

Generate text (summaries, replies, translations).

Approaches:
- **Encoder-decoder**: T5, BART.
- **Decoder-only**: GPT-style models.
- **Conditional generation**: prompt + model.

### Summarization

Condense a document into a shorter version.

- **Extractive**: pick the most important sentences.
- **Abstractive**: generate new text (encoder-decoder or decoder-only).

Evaluation: ROUGE (overlap with reference), BERTScore, human eval.

## Evaluation Metrics for NLP

| Task | Metric |
|---|---|
| Classification | Accuracy, F1, PR-AUC |
| NER | F1 on entity-level matches |
| Translation | BLEU, chrF |
| Summarization | ROUGE, BERTScore |
| Generation | Human eval, perplexity, BLEU/ROUGE |
| Retrieval | Recall@k, MRR, NDCG |
| Similarity | Pearson/Spearman with human ratings |

For LLM-based systems, human evaluation is the gold standard. Automated metrics (BLEU, ROUGE) have known limitations.

## The Modern NLP Stack

A typical 2024+ NLP pipeline:

1. **Embedding model**: sentence-transformers, OpenAI embeddings, or open-source.
2. **Vector store**: FAISS, Pinecone, Weaviate, Qdrant, pgvector.
3. **LLM**: GPT-4, Claude, Llama, Mistral, etc.
4. **Orchestration**: LangChain, LlamaIndex, custom code.
5. **Evaluation**: human eval, automated metrics, LLM-as-judge.

For RAG (retrieval-augmented generation):
1. User query → embedding model → vector.
2. Search vector store for relevant documents.
3. Pass query + retrieved context to LLM.
4. LLM generates answer.
5. Evaluate retrieval (recall@k) and generation (faithfulness, relevance).

## Fine-Tuning vs Prompting

For NLP tasks, the choice is:

**Prompting (zero-shot/few-shot)**:
- Use a pretrained LLM as-is.
- Provide instructions and/or examples in the prompt.
- No training needed.
- Works well for many tasks, especially general ones.
- Can be expensive at scale (per-token cost).

**Fine-tuning**:
- Take a pretrained model and train on task-specific data.
- Higher performance on specific tasks.
- More upfront cost, lower per-request cost.
- Requires labeled data and ML infrastructure.

**Hybrid**:
- Use a smaller fine-tuned model for cheap, high-volume predictions.
- Use a large LLM for complex, low-volume cases.

## What to Study for NLP Interviews

The job description lists NLP as "desired," not required. Focus on:

1. **Tokenization**: word, subword (BPE), SentencePiece.
2. **TF-IDF and classical features**: still a strong baseline.
3. **Word embeddings**: Word2Vec, GloVe, FastText conceptually.
4. **Transformer basics**: self-attention, BERT vs GPT.
5. **Fine-tuning BERT for classification**: the most common interview task.
6. **Text classification metrics**: F1, PR-AUC for imbalanced classes.
7. **RAG basics**: retrieval + generation, vector databases.
8. **Evaluation**: task-appropriate metrics, human eval considerations.

You do not need to know:
- Cutting-edge LLM architecture details.
- Training LLMs from scratch.
- Multilingual NLP internals.

But you do need to be able to discuss:
- When to use classical vs transformer approaches.
- How to evaluate text-based models.
- How to handle class imbalance in text classification.
- How to deploy and monitor text models.
