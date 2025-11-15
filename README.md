# Ubuntu Dialogue Corpus - Sentiment Analysis & Topic Classification Pipeline

Pipeline for sentiment analysis and topic extraction from the Ubuntu Dialogue Corpus dataset using VADER sentiment analysis, BERTopic, and DuckDB streaming.

## Features

- 📊 **Complete Analysis**: 346K+ conversations with sentiment scores and topic classification
- 🔬 **Evaluation-Focused**: Designed for speed and easy evaluation rather than production deployment
- 🏷️ **Topic Extraction**: BERTopic with semantic alignment to predefined categories
- 🔍 **Inspection Tools**: Evaluate topic mappings before final alignment


## Design Philosophy

This pipeline is intentionally designed with **speed and easy evaluation** in mind. 

It prioritizes:

- ✅ Quick setup and execution
- ✅ Minimal dependencies
- ✅ Easy to understand and modify
- ✅ Fast iteration for experimentation

### Production Considerations

For production deployment, consider adding:
- **Docker**: Containerization for consistent environments and easy deployment
- **Orchestration**: DAG-based workflow managers like Airflow, Prefect, or Dagster for scheduling, monitoring, and retry logic
- **Configuration Management**: Environment variables, config files, and secrets management
- **Monitoring**: Logging, metrics, and alerting systems

## Getting Started

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) - a fast Python package installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup & Installation

```bash
# Clone repository
git clone https://github.com/alex-kolmakov/text-sentiment-and-topic-classification.git
cd text-sentiment-and-topic-classification

# Install dependencies
uv pip install -r requirements.txt
```

**Dependencies installed:**
- `kagglehub` - Download Ubuntu Dialogue Corpus
- `vaderSentiment` - Fast rule-based sentiment analysis
- `duckdb` - Streaming large CSV files efficiently
- `bertopic` - Topic modeling with transformers
- `sentence-transformers` - Semantic embeddings for topic alignment
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation

### Run the Complete Pipeline

```bash
# Step 1: Download dataset (~799MB, cached locally)
uv run download_data.py

# Step 2: Sentiment analysis (~2 minutes)
uv run sentiment_pipeline.py

# Step 3: Topic extraction & alignment (~10 minutes)
uv run topic_pipeline.py
```

### Query Results

```bash
# Sentiment distribution
duckdb conversations.duckdb -c "SELECT sentiment, COUNT(*) FROM conversations_with_sentiment GROUP BY sentiment"

# Topic distribution
duckdb conversations.duckdb -c "SELECT aligned_topic, COUNT(*), ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct FROM conversations_with_topics WHERE aligned_topic IS NOT NULL GROUP BY aligned_topic ORDER BY COUNT(*) DESC"

# Topic mapping inspection
duckdb conversations.duckdb -box -c "SELECT extracted_topic_id as id, LEFT(extracted_keywords, 60) as keywords, aligned_topic, ROUND(confidence, 3) as conf, num_conversations as count FROM topic_mapping_inspection ORDER BY num_conversations DESC LIMIT 15"
```


## How It Works

### 1. Data Loading & Aggregation

**Input:** Ubuntu Dialogue Corpus (1,038,324 dialogue messages)

**Process:** DuckDB aggregates messages into complete conversations:

```sql
CREATE TABLE conversations AS
SELECT 
    dialogueID,
    STRING_AGG(text, ' ') as conversation_text
FROM read_csv_auto('dialogueText.csv')
WHERE text IS NOT NULL
GROUP BY dialogueID
```

**Output:** 346,108 complete conversations

**Why conversation-level?**
- Context matters for sentiment
- Holistic view of technical support interactions
- Matches output specification (one row per conversation)

### 2. Sentiment Analysis (VADER)

**Process:**
- Analyzes each complete conversation (not individual messages)
- Processes in 50,000-row batches for memory efficiency
- VADER calculates compound score: -1 (negative) to +1 (positive)

**Classification:**
- **POSITIVE**: compound ≥ 0.05
- **NEGATIVE**: compound ≤ -0.05
- **NEUTRAL**: -0.05 < compound < 0.05

**Runtime:** ~2 minutes (3,000-15,000 conversations/second)

### 3. Topic Extraction (BERTopic)

**Step 3a: Extract Topics**
- Embeds conversations using sentence transformers (`all-MiniLM-L6-v2`)
- Reduces dimensionality with UMAP
- Clusters similar conversations with HDBSCAN
- Extracts keywords using c-TF-IDF
- Filters stop words ('the', 'to', 'get', 'it', chat terms)

**Step 3b: Align to Predefined Categories**

Predefined technical support categories:
- **Hardware Issues** - Devices, drivers, peripherals, disk, graphics
- **Software Installation** - Packages, apt-get, dependencies, compilation
- **Network Configuration** - WiFi, DNS, firewall, router, IP addresses
- **User Permissions** - sudo, chmod, access denied, authentication
- **System Performance** - Slow, CPU, memory, freeze, optimization
- **Data Recovery** - Backup, restore, lost files, corrupted filesystem
- **Pizzas with Ketchup** ⚠️ - Outlier detection test topic
- **Dog Walking** ⚠️ - Outlier detection test topic
- **Climbing Mountains** ⚠️ - Outlier detection test topic

**Alignment Process:**
1. Encode extracted topic keywords using sentence transformers
2. Encode predefined topic descriptions (rich keyword sets)
3. Calculate cosine similarity between extracted ↔ predefined
4. Assign best matching category if confidence > 0.19
5. Otherwise classify as "Other/Unclassified"


### 4. Results Storage (DuckDB)

All results saved in `conversations.duckdb`:

**Table: conversations** - Raw aggregated conversations (346K rows)

**Table: conversations_with_sentiment** - + Sentiment analysis
- `sentiment` - POSITIVE/NEGATIVE/NEUTRAL classification
- `compound_score` - Overall sentiment score (-1 to 1)
- `pos`, `neg`, `neu` - Component sentiment scores

**Table: conversations_with_topics** - + Topic extraction & alignment
- `extracted_topic_id` - BERTopic cluster ID
- `extracted_topic_keywords` - Raw keywords from BERTopic
- `aligned_topic` - Predefined category assignment
- `alignment_confidence` - Similarity score (0-1)

**Table: topic_mapping_inspection** - Evaluation table
- Shows extracted topic → aligned category mapping
- Confidence scores and conversation counts per topic
- Use for evaluating alignment quality

> **Note**: Tables between sentiment analysis and topic extraction intentionally separated for ability to work and evaluate them separately.

## Project Structure

```
text-sentiment-analysis/
├── README.md                      # This file
├── ONLINE_TOPIC_MODELING.md       # Advanced topic modeling approaches
├── requirements.txt               # Python dependencies
├── download_data.py               # Download Ubuntu Dialogue Corpus
├── sentiment_pipeline.py          # Sentiment analysis pipeline
├── topic_pipeline.py              # Topic extraction & alignment
└── conversations.duckdb           # Output database
    ├── conversations                    # 346K aggregated conversations
    ├── conversations_with_sentiment     # + sentiment analysis
    ├── conversations_with_topics        # + topic extraction
    └── topic_mapping_inspection         # Topic alignment evaluation
```

---

## Technologies Used

- **[VADER Sentiment](https://github.com/cjhutto/vaderSentiment)** - Rule-based sentiment analysis optimized for social media
- **[BERTopic](https://maartengr.github.io/BERTopic/)** - Topic modeling with transformers
- **[Sentence Transformers](https://www.sbert.net/)** - Semantic embeddings for text
- **[DuckDB](https://duckdb.org/)** - In-process SQL database for streaming analytics
- **[kagglehub](https://github.com/Kaggle/kagglehub)** - Kaggle dataset download API
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package installer

---

## Dataset

**Ubuntu Dialogue Corpus**
- **Source**: [Kaggle - Ubuntu Dialogue Corpus](https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus)
- **Size**: ~799MB compressed
- **Rows**: 1,038,324 dialogue turns
- **Conversations**: 346,108 unique dialogues
- **Content**: Technical support conversations from Ubuntu IRC channels
- **License**: Creative Commons Attribution-ShareAlike 3.0 Unported License

---

## Quick Reference Commands

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup
git clone https://github.com/alex-kolmakov/text-sentiment-and-topic-classification.git
cd text-sentiment-and-topic-classification
uv pip install -r requirements.txt

# Run pipeline
uv run download_data.py
uv run sentiment_pipeline.py
uv run topic_pipeline.py

# Query results
duckdb conversations.duckdb -c "SELECT sentiment, COUNT(*) FROM conversations_with_sentiment GROUP BY sentiment"
duckdb conversations.duckdb -c "SELECT aligned_topic, COUNT(*) FROM conversations_with_topics WHERE aligned_topic IS NOT NULL GROUP BY aligned_topic ORDER BY COUNT(*) DESC"
```

## Notes on Online Topic Modeling

For evolving topic categories without manual predefinition, see [ONLINE_TOPIC_MODELING.md](ONLINE_TOPIC_MODELING.md) which covers:
- Pure discovery (fully automatic)
- Semi-supervised (example-guided)
- Hierarchical clustering
- Incremental learning for streaming data
- Hybrid approach (recommended for production)


## License

This project is licensed under the MIT License. The Ubuntu Dialogue Corpus dataset is licensed under Creative Commons Attribution-ShareAlike 3.0.
