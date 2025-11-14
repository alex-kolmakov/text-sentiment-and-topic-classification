#!/usr/bin/env python3
"""
Ubuntu Dialogue Corpus Sentiment Analysis Pipeline

Aggregates multiple dialogue messages into conversations and performs sentiment analysis.
Each conversation (dialogueID) contains multiple turns/replies that are combined into
a single text for sentiment scoring.

Features:
- VADER sentiment analysis (fast, no model downloads)  
- DuckDB for streaming CSV data and storing results (no memory overload)
- Simple sequential processing (VADER is already fast!)
- Output: DuckDB database with one row per conversation

Run: uv run sentiment_pipeline.py
"""

import duckdb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path
import time

print("=" * 70)
print("Ubuntu Dialogue Corpus - Sentiment Analysis Pipeline")
print("=" * 70)
print()

# Step 1: Use cached dataset
print("📥 Step 1: Locating dataset...")
csv_file = Path.home() / ".cache/kagglehub/datasets/rtatman/ubuntu-dialogue-corpus/versions/2/Ubuntu-dialogue-corpus/dialogueText.csv"

if not csv_file.exists():
    print(f"❌ Dataset not found at: {csv_file}")
    print("Please run download_data.py first to download the dataset.")
    exit(1)

print(f"✓ Dataset found")
print()

# Step 2: Setup DuckDB
print("📂 Step 2: Setting up DuckDB...")
output_db = "conversations.duckdb"
conn = duckdb.connect(output_db)

# Create CSV view and structure conversations (all in DuckDB - no memory issues!)
print("  Processing CSV and structuring conversations...")

conn.execute(f"""
    CREATE OR REPLACE TABLE conversations AS
    SELECT 
        dialogueID,
        STRING_AGG(text, ' ') as conversation_text
    FROM read_csv_auto('{csv_file}')
    WHERE text IS NOT NULL
    GROUP BY dialogueID
""")

conversation_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
print(f"✓ Structured {conversation_count:,} conversations")
print()

# Step 3: Analyze sentiment with VADER
print("💭 Step 3: Analyzing sentiment with VADER...")
analyzer = SentimentIntensityAnalyzer()

# Create results table (replace if exists from previous run)
conn.execute("""
    CREATE OR REPLACE TABLE conversations_with_sentiment (
        dialogueID VARCHAR,
        conversation_text VARCHAR,
        sentiment VARCHAR,
        sentiment_score DOUBLE,
        pos_score DOUBLE,
        neg_score DOUBLE,
        neu_score DOUBLE
    )
""")

# Process in chunks to show progress
chunk_size = 25000
offset = 0
start_time = time.time()

while offset < conversation_count:
    # Fetch chunk
    chunk = conn.execute(f"""
        SELECT dialogueID, conversation_text 
        FROM conversations 
        LIMIT {chunk_size} OFFSET {offset}
    """).fetchall()
    
    if not chunk:
        break
    
    # Analyze sentiment for each conversation in chunk
    results = []
    for dialogue_id, text in chunk:
        if text and len(text.strip()) > 0:
            scores = analyzer.polarity_scores(text)
            
            # Classify based on compound score
            if scores['compound'] >= 0.05:
                sentiment = 'POSITIVE'
            elif scores['compound'] <= -0.05:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
            
            results.append((
                dialogue_id, text, sentiment,
                scores['compound'], scores['pos'], scores['neg'], scores['neu']
            ))
        else:
            results.append((
                dialogue_id, text, 'NEUTRAL',
                0.0, 0.0, 0.0, 0.0
            ))
    
    # Insert batch into DuckDB
    conn.executemany("""
        INSERT INTO conversations_with_sentiment 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, results)
    
    offset += chunk_size
    elapsed = time.time() - start_time
    progress = (offset / conversation_count) * 100
    rate = offset / elapsed if elapsed > 0 else 0
    
    print(f"  Progress: {min(offset, conversation_count):,}/{conversation_count:,} ({min(progress, 100):.1f}%) - {rate:.0f} conv/sec")

elapsed = time.time() - start_time
print(f"✓ Completed in {elapsed:.1f} seconds")
print()

# Summary
print("=" * 70)
print("✅ PIPELINE COMPLETED!")
print("=" * 70)
print(f"\nOutput database:")
print(f"  • {output_db}")
print(f"\nTables:")
print(f"  • conversations (raw aggregated conversations)")
print(f"  • conversations_with_sentiment (with sentiment analysis)")
print(f"\nStats:")
print(f"  • Processed: {conversation_count:,} conversations")
print(f"  • Time: {elapsed:.1f} seconds")
print(f"  • Speed: {conversation_count/elapsed:.0f} conversations/second")

# Sentiment distribution
print(f"\nSentiment distribution:")
sentiment_counts = conn.execute("""
    SELECT sentiment, COUNT(*) as count 
    FROM conversations_with_sentiment 
    GROUP BY sentiment 
    ORDER BY count DESC
""").fetchall()

for sentiment, count in sentiment_counts:
    percentage = (count / conversation_count) * 100
    print(f"  • {sentiment}: {count:,} ({percentage:.1f}%)")