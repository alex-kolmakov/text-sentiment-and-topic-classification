# Ubuntu Dialogue Corpus - Sentiment Analysis Pipeline

A fast and memory-efficient sentiment analysis pipeline for the Ubuntu Dialogue Corpus dataset using VADER sentiment analysis and DuckDB streaming.

## Features

- 🚀 **Fast Processing**: Uses VADER sentiment analysis (processes thousands of conversations per second)
- 💾 **Memory Efficient**: DuckDB streaming handles 1M+ rows without loading entire dataset into memory
- 📊 **Complete Analysis**: Analyzes 346K+ conversations with sentiment scores and classifications
- 🎯 **Simple**: Sequential processing - no complex multiprocessing or model downloads required
- 🔬 **Evaluation-Focused**: Designed for speed and easy evaluation rather than production deployment

## Design Philosophy

This pipeline is intentionally designed with **speed and easy evaluation** in mind. It prioritizes:

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
- **Error Handling**: Robust retry mechanisms and failure notifications
- **Data Validation**: Schema validation and data quality checks
- **Scalability**: Distributed processing (Spark, Dask) for larger datasets
- **CI/CD**: Automated testing and deployment pipelines

This simplified approach allows for rapid development and testing while keeping the codebase accessible and maintainable for evaluation purposes.

## Getting Started on a New Machine

Follow these steps to set up and run the pipeline on a fresh machine:

### Step 1: Install uv (Python Package Manager)

First, [install `uv`](https://docs.astral.sh/uv/getting-started/installation/) - a fast Python package installer and resolver.


### Step 2: Clone the Repository

```bash
git clone https://github.com/alex-kolmakov/text-sentiment-analysis.git
cd text-sentiment-analysis
```

### Step 3: Install Python Dependencies

`uv` will automatically create a virtual environment and install all required packages:

```bash
uv pip install -r requirements.txt
```

This installs primarily:
- `kagglehub` - For downloading the Ubuntu Dialogue Corpus
- `vaderSentiment` - Fast sentiment analysis
- `duckdb` - Streaming large CSV files

### Step 4: Download the Dataset once
Download the Ubuntu Dialogue Corpus dataset (~799MB) before running the pipeline:

```bash
uv run download_data.py
```

This will:
- Download the dataset from Kaggle via kagglehub
- Cache it locally at: `~/.cache/kagglehub/datasets/rtatman/ubuntu-dialogue-corpus/`
- Verify the dataset is ready for processing

### Step 5: Run the Sentiment Analysis Pipeline

```bash
uv run sentiment_pipeline.py
```

The pipeline will:
1. Locate the cached dataset
2. Structure 1M+ dialogue turns into 346K conversations
3. Analyze sentiment for each conversation using VADER
4. Save results to `conversations.duckdb`

Expected runtime: **~2 minutes** on modern hardware (processing ~3,000-15,000 conversations/second)

### Step 6: Query the Results

Use DuckDB to explore the results:

```bash
uv run python -c "
import duckdb
conn = duckdb.connect('conversations.duckdb')
print(conn.execute('SELECT sentiment, COUNT(*) FROM conversations_with_sentiment GROUP BY sentiment').fetchdf())
"
```

That's it! You now have a complete sentiment analysis database ready for exploration.

---

## Quick Start (TL;DR)

For experienced users, here's the minimal command sequence:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/alex-kolmakov/text-sentiment-analysis.git
cd text-sentiment-analysis
uv pip install -r requirements.txt

# Download dataset
uv run download_data.py

# Run pipeline
uv run sentiment_pipeline.py
```

---

## How It Works

### Pipeline Steps

1. **Data Loading**: Locates the Ubuntu Dialogue Corpus CSV from kagglehub (1,038,324 individual dialogue messages)

2. **Conversation Aggregation**: Uses DuckDB's `STRING_AGG` function to combine multiple messages into complete conversations
   ```sql
   CREATE TABLE conversations AS
   SELECT 
       dialogueID,
       STRING_AGG(text, ' ') as conversation_text
   FROM read_csv_auto('dialogueText.csv')
   WHERE text IS NOT NULL
   GROUP BY dialogueID
   ```
   
   **How STRING_AGG Works:**
   - Groups all messages by `dialogueID` (each conversation has a unique ID)
   - Concatenates message texts with a **single space** (`' '`) as the separator
   - Preserves message order within each conversation
   - Handles NULL/empty messages gracefully
   
   **Example:**
   - Message 1: `"Hello folks, please help me"`
   - Message 2: `"Did I choose a bad channel?"`
   - Message 3: `"the second sentence is better english"`
   - **Result**: `"Hello folks, please help me Did I choose a bad channel? the second sentence is better english"`


3. **Sentiment Analysis**: Processes complete conversations (not individual messages) in 50,000-row chunks using VADER
   - Each conversation is analyzed as a single text unit
   - VADER calculates compound score and sentiment components
   - Sentiment classification based on compound score thresholds

4. **Results Storage**: Saves all results directly to DuckDB database with one row per conversation

### Sentiment Classification

VADER produces a compound score from -1 (most negative) to +1 (most positive):
- **POSITIVE**: compound score ≥ 0.05
- **NEGATIVE**: compound score ≤ -0.05
- **NEUTRAL**: compound score between -0.05 and 0.05

### Why Conversation-Level (Not Message-Level)?

The pipeline analyzes **complete conversations** rather than individual messages because:
- Context matters: A negative reply might be addressing a positive question
- Holistic sentiment: The overall tone of a conversation is more meaningful than individual messages
- Reduces noise: Individual messages can be ambiguous without conversational context
- Matches use case: Technical support conversations are best evaluated as complete interactions

## Project Structure

```
text-sentiment-analysis/
├── README.md                  # This file - complete documentation
├── requirements.txt           # Python dependencies (kagglehub, vaderSentiment, duckdb, etc.)
├── download_data.py           # Optional: Download Ubuntu Dialogue Corpus dataset
├── sentiment_pipeline.py      # Main pipeline: aggregates conversations & analyzes sentiment
└── conversations.duckdb       # Output: DuckDB database with results (generated after running pipeline)
```

### File Descriptions

- **`download_data.py`**: Downloads the Ubuntu Dialogue Corpus from Kaggle via kagglehub and caches it locally. Run this once before the pipeline, or let the pipeline check for the dataset automatically.

- **`sentiment_pipeline.py`**: Main pipeline that:
  - Locates the cached dataset
  - Uses DuckDB to aggregate 1M+ messages into 346K conversations with `STRING_AGG`
  - Analyzes sentiment for each complete conversation using VADER
  - Stores results in DuckDB database

- **`conversations.duckdb`**: Generated output database containing two tables:
  - `conversations`: Aggregated conversation texts
  - `conversations_with_sentiment`: Final results with sentiment scores

## Technologies Used

- **[VADER Sentiment](https://github.com/cjhutto/vaderSentiment)**: Rule-based sentiment analysis optimized for social media text
- **[DuckDB](https://duckdb.org/)**: In-process SQL OLAP database for streaming large datasets
- **[kagglehub](https://github.com/Kaggle/kagglehub)**: Kaggle dataset download API
- **[uv](https://github.com/astral-sh/uv)**: Fast Python package installer and resolver

## Dataset

**Ubuntu Dialogue Corpus**
- Source: [Kaggle](https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus)
- Size: ~799MB compressed
- Rows: 1,038,324 dialogue turns
- Conversations: 346,108 unique dialogues
- Content: Technical support conversations from Ubuntu IRC channels