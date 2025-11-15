#!/usr/bin/env python3
"""
Ubuntu Dialogue Corpus Topic Extraction and Alignment Pipeline

Uses BERTopic to extract topics from conversations and aligns them with
predefined topics relevant to technical support context.

Predefined Topics:
- Hardware Issues
- Software Installation
- Network Configuration
- Pizzas with Ketchup
- User Permissions
- Dog Walking
- System Performance
- Climbing Mountains
- Data Recovery

Run: uv run topic_pipeline.py
"""

import os
# Disable tokenizer parallelism warning (must be set before importing transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import duckdb
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import time
from collections import defaultdict

print("=" * 70)
print("Ubuntu Dialogue Corpus - Topic Extraction & Alignment Pipeline")
print("=" * 70)
print()

# ============================================================================
# PREDEFINED TOPICS FOR ALIGNMENT
# ============================================================================
# Each topic includes rich keyword descriptions for better semantic matching
# with extracted topics from BERTopic
# ============================================================================

PREDEFINED_TOPICS = {
    "Hardware Issues": (
        "hardware devices components drivers printer scanner mouse keyboard "
        "cuda gpu nvida graphics card amd radeon bluetooth pulseaudio adapter "
        "touchpad monitor screen display graphics card sound audio speakers "
        "microphone webcam camera usb ports disk drive ssd hdd storage "
        "motherboard bios uefi firmware crash failure malfunction error broken "
        "not working detected recognized connected physical device peripherals"
    ),
    
    "Software Installation": (
        "install installation setup configure package packages repository repositories "
        "apt apt-get dpkg snap flatpak ppa dependencies dependency driver library libraries "
        "compile compilation build make cmake gcc version versions upgrade update download manager"
        "uninstall remove purge software application program applications programs mozilla "
        "missing package manager synaptic gdebi error failed cannot find browser firefox chrome"
    ),
    
    "Network Configuration": (
        "network networking wifi wireless ethernet wired connection connectivity "
        "internet online offline ip address ipv4 ipv6 dhcp static dns nameserver "
        "gateway subnet mask router modem access point firewall ufw iptables "
        "port ports ssh ftp http https proxy vpn ping traceroute netstat ifconfig "
        "network-manager nm-applet connected disconnected cannot connect"
    ),
    
    "User Permissions": (
        "permission access denied forbidden privilege cannot "
        "sudo su root superuser administrator admin user group "
        "chmod chown chgrp owner ownership read write execute rwx file "
        "directory folder folders rights authority access "
        "not allowed authentication password login"
    ),
    
    "System Performance": (
        "performance speed fast faster optimization "
        "memory ram swap cpu processor usage utilization load heavy freeze "
        "hang unresponsive lagging lag black screen blue screen"
        "delay delayed boot startup shutdown restart time long resource "
        "resources bottleneck throttle high consumption"
    ),
    
    "Data Recovery": (
        "hdd harddrive ssd usb flashrecover backup restore lost missing deleted "
        "removed deleted data documents partition "
        "partitions corrupted damaged broken repair fix filesystem ext4 ntfs "
        "fat32 format formatted accidentally testdisk photorec timeshift rsync save"
    ),

    "Pizzas with Ketchup": (
        "pizza pizzas ketchup tomato sauce food restaurant eating meal "
        "dinner lunch cheese pepperoni topping delicious tasty hungry"
    ),
    
    "Dog Walking": (
        "dog dogs walk walking pet pets animal animals puppy puppies "
        "outside outdoor park leash collar exercise training"
    ),
    
    "Climbing Mountains": (
        "mountain mountains climbing climb hiking hike summit peak altitude "
        "trail trails outdoor adventure nature elevation rope gear equipment"
    )
}

# Extract as lists to maintain consistent index alignment between names and descriptions
# This ensures TOPIC_NAMES[i] always corresponds to TOPIC_DESCRIPTIONS[i] for matching
TOPIC_NAMES = list(PREDEFINED_TOPICS.keys())
TOPIC_DESCRIPTIONS = list(PREDEFINED_TOPICS.values())

# Step 1: Check if sentiment analysis was run
print("📥 Step 1: Loading conversations from database...")
db_file = "conversations.duckdb"

if not Path(db_file).exists():
    print(f"❌ Database not found: {db_file}")
    print("Please run sentiment_pipeline.py first to create the database.")
    exit(1)

conn = duckdb.connect(db_file)

# Check if we have sentiment data
try:
    count = conn.execute("SELECT COUNT(*) FROM conversations_with_sentiment").fetchone()[0]
    print(f"✓ Found {count:,} conversations with sentiment analysis")
except:
    print("❌ Table 'conversations_with_sentiment' not found.")
    print("Please run sentiment_pipeline.py first.")
    exit(1)

print()

# Step 2: Load sentence transformer for topic alignment
print("🤖 Step 2: Loading embedding model for topic alignment...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded")

# Encode predefined topic descriptions for similarity matching
print("  Encoding predefined topic descriptions...")
predefined_embeddings = embedding_model.encode(TOPIC_DESCRIPTIONS, convert_to_tensor=True)
print(f"✓ Encoded {len(TOPIC_NAMES)} predefined topics")
print()

# Step 3: Extract topics using BERTopic
print("📊 Step 3: Extracting topics with BERTopic...")
print("  This may take several minutes for 346K conversations...")
print()

# === CONFIGURATION ===
# Expected runtime for full dataset: 10-15 minutes
# Feel free to adjust settings for your environment
# ====================

# Load conversations (limit for faster processing in demo)
conversations_data = conn.execute(f"""
    SELECT dialogueID, conversation_text 
    FROM conversations_with_sentiment
    WHERE conversation_text IS NOT NULL 
    AND LENGTH(conversation_text) > 20
""").fetchall()

dialogue_ids = [row[0] for row in conversations_data]
conversations = [row[1] for row in conversations_data]

print(f"  Loaded {len(conversations):,} conversations")
print("  Training BERTopic model...")
start_time = time.time()

# Configure vectorizer to remove stop words and meaningless terms
# This filters out common words like 'the', 'to', 'get', 'it', etc.
# Also add custom stop words relevant to chat conversations
custom_stop_words = [
    'just', 'like', 'know', 'want', 'need', 'think', 'really', 'yeah', 'yes', 'ok',
    'oh', 'ah', 'hmm', 'uh', 'got', 'get', 'thing', 'things', 'stuff', 'way',
    'going', 'want', 'make', 'use', 'using', 'used', 'does', 'did', 'doing', 'the', 'to',
    'thanks', 'thank', 'please', 'help', 'hi', 'hello', 'hey'
]

# Combine with English stop words - convert to list
from sklearn.feature_extraction import text
stop_words = list(text.ENGLISH_STOP_WORDS.union(custom_stop_words))

vectorizer_model = CountVectorizer(
    stop_words=stop_words,  # Remove stop words (English + custom) as list
    min_df=15,  # Word must appear in at least 15 documents
    max_df=0.95,  # Ignore words appearing in more than 95% of documents
    ngram_range=(1, 2),  # Use unigrams and bigrams
    max_features=1000  # Limit vocabulary size for efficiency
)

# Configure BERTopic with optimized settings
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,  # Use custom vectorizer with stop words
    min_topic_size=200,
    nr_topics=100,
    calculate_probabilities=False,
    verbose=True
)

# Fit the model and extract topics
topics, probabilities = topic_model.fit_transform(conversations)

elapsed = time.time() - start_time
print(f"✓ Topic extraction completed in {elapsed:.1f} seconds")
print()

# Get topic information
topic_info = topic_model.get_topic_info()
print(f"  Discovered {len(topic_info) - 1} topics (excluding outliers)")
print()

# Step 4: Align extracted topics with predefined topics
print("🔗 Step 4: Aligning extracted topics with predefined categories...")

# Get topic representations
topic_labels = {}
topic_alignments = {}

for topic_id in topic_info['Topic'].values:
    if topic_id == -1:  # Skip outlier topic
        continue
    
    # Get top words for this topic
    topic_words = topic_model.get_topic(topic_id)
    if topic_words:
        # Create a topic representation from top words (use more words for better matching)
        topic_representation = " ".join([word for word, _ in topic_words[:20]])
        topic_labels[topic_id] = topic_representation
        
        # Encode the topic representation
        topic_embedding = embedding_model.encode(topic_representation, convert_to_tensor=True)
        
        # Calculate similarity with predefined topic descriptions
        similarities = util.cos_sim(topic_embedding, predefined_embeddings)[0]
        
        # Get best matching predefined topic
        best_match_idx = similarities.argmax().item()
        best_match_score = similarities[best_match_idx].item()
        
        # Lower threshold since we're using richer descriptions
        if best_match_score > 0.19:
            topic_alignments[topic_id] = {
                'aligned_topic': TOPIC_NAMES[best_match_idx],
                'confidence': best_match_score
            }
        else:
            topic_alignments[topic_id] = {
                'aligned_topic': 'Other/Unclassified',
                'confidence': best_match_score
            }

print(f"✓ Aligned {len(topic_alignments)} extracted topics to predefined categories")
print()

# Step 5: Create final table with topic information
print("💾 Step 5: Creating final table with topic information...")

# Build a dictionary mapping dialogueID to topic information
print(f"  Processing {len(dialogue_ids):,} conversations...")
topic_data = {}

for i, dialogue_id in enumerate(dialogue_ids):
    topic_id = topics[i]
    
    if topic_id == -1:
        # Outlier topic
        topic_data[dialogue_id] = {
            'extracted_topic_id': topic_id,
            'extracted_topic_keywords': 'outlier',
            'aligned_topic': 'Other/Unclassified',
            'alignment_confidence': 0.0
        }
    else:
        # Regular topic
        alignment = topic_alignments.get(topic_id, {'aligned_topic': 'Other/Unclassified', 'confidence': 0.0})
        topic_keywords = topic_labels.get(topic_id, "unknown")
        
        topic_data[dialogue_id] = {
            'extracted_topic_id': topic_id,
            'extracted_topic_keywords': topic_keywords,
            'aligned_topic': alignment['aligned_topic'],
            'alignment_confidence': alignment['confidence']
        }

# Create temporary table with topic data
print("  Creating temporary table with topic data...")

topic_df = pd.DataFrame([
    {
        'dialogueID': dialogue_id,
        'extracted_topic_id': data['extracted_topic_id'],
        'extracted_topic_keywords': data['extracted_topic_keywords'],
        'aligned_topic': data['aligned_topic'],
        'alignment_confidence': data['alignment_confidence']
    }
    for dialogue_id, data in topic_data.items()
])

# Register the DataFrame as a temporary table
conn.register('temp_topics', topic_df)

# Create final table by joining sentiment data with topics in one operation
print("  Creating final table with JOIN...")
conn.execute("""
    CREATE OR REPLACE TABLE conversations_with_topics AS
    SELECT 
        s.*,
        COALESCE(t.extracted_topic_id, NULL) AS extracted_topic_id,
        COALESCE(t.extracted_topic_keywords, NULL) AS extracted_topic_keywords,
        COALESCE(t.aligned_topic, NULL) AS aligned_topic,
        COALESCE(t.alignment_confidence, NULL) AS alignment_confidence
    FROM conversations_with_sentiment s
    LEFT JOIN temp_topics t ON s.dialogueID = t.dialogueID
""")

# Clean up temporary table
conn.unregister('temp_topics')

print("✓ Final table created")
print()

# Calculate topic counts for inspection
topic_counts = defaultdict(int)
for topic_id in topics:
    if topic_id != -1:
        topic_counts[topic_id] += 1

# Step 6: Save topic mapping for inspection
print("📋 Step 6: Creating topic mapping table for inspection...")

# Create a table showing the mapping between extracted topics and aligned topics
topic_mapping_data = []
for topic_id, keywords in topic_labels.items():
    alignment = topic_alignments.get(topic_id, {'aligned_topic': 'Other/Unclassified', 'confidence': 0.0})
    count = topic_counts.get(topic_id, 0)
    
    topic_mapping_data.append({
        'extracted_topic_id': topic_id,
        'extracted_keywords': keywords,
        'aligned_topic': alignment['aligned_topic'],
        'confidence': alignment['confidence'],
        'num_conversations': topic_counts.get(topic_id, 0)
    })

# Create inspection table
mapping_df = pd.DataFrame(topic_mapping_data)
mapping_df = mapping_df.sort_values('num_conversations', ascending=False)

conn.execute("""
    CREATE OR REPLACE TABLE topic_mapping_inspection AS
    SELECT * FROM mapping_df
""")

print(f"✓ Created topic_mapping_inspection table with {len(mapping_df)} topic mappings")
print()

print("=" * 70)
print("✅ PIPELINE COMPLETED!")
print("=" * 70)

# Step 7: Generate insights
# Show topic mapping for inspection
print("\n🔍 TOPIC MAPPING INSPECTION (Extracted → Aligned):")
print("=" * 70)
print("\nTop 10 Extracted Topics and Their Alignments:\n")

for idx, row in mapping_df.head(10).iterrows():
    print(f"Topic #{row['extracted_topic_id']}:")
    print(f"  Extracted Keywords: {row['extracted_keywords'][:80]}")
    print(f"  → Aligned to: {row['aligned_topic']}")
    print(f"  Confidence: {row['confidence']:.3f}")
    print(f"  Conversations: {row['num_conversations']:,}")
    print()

print("=" * 70)

# Topic distribution
print("\nExtracted Topic Distribution (Top 10):")

sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for topic_id, count in sorted_topics:
    keywords = topic_labels.get(topic_id, "unknown")[:60]
    percentage = (count / len(topics)) * 100
    print(f"  Topic {topic_id}: {count:,} ({percentage:.1f}%) - {keywords}...")

# Sample results showing extraction and alignment - one from each category
print("\n📝 Sample Conversations (Extraction → Alignment):")
print("=" * 70)
print("Showing one example from each aligned category:\n")

# Get one sample from each aligned category
samples = conn.execute("""
    WITH ranked_samples AS (
        SELECT 
            dialogueID,
            LEFT(conversation_text, 100) as preview,
            sentiment,
            extracted_topic_keywords,
            aligned_topic,
            alignment_confidence,
            ROW_NUMBER() OVER (PARTITION BY aligned_topic ORDER BY alignment_confidence DESC) as rn
        FROM conversations_with_topics
        WHERE aligned_topic IS NOT NULL
    )
    SELECT dialogueID, preview, sentiment, extracted_topic_keywords, aligned_topic, alignment_confidence
    FROM ranked_samples
    WHERE rn = 1
    ORDER BY alignment_confidence DESC
""").fetchall()

if not samples:
    print("  No aligned topics found (all conversations classified as Other/Unclassified)")
else:
    for i, (did, text, sent, extracted, aligned, conf) in enumerate(samples, 1):
        # Split extracted keywords into list for display
        keywords_list = extracted.split()[:5]  # Show top 5 keywords
        
        print(f"{i}. Chat #{did}")
        print(f"   Text: {text}...")
        print(f"   Sentiment: [{sent}]")
        print(f"   → extracted_topics: {keywords_list}")
        print(f"   → aligned_topic: \"{aligned}\" (confidence: {conf:.3f})")
        print()

print("\n" + "=" * 70)

# Show aligned topic distribution
print("\n📊 Aligned Topic Distribution:")
distribution = conn.execute("""
    SELECT 
        aligned_topic, 
        COUNT(*) as count, 
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM conversations_with_topics 
    WHERE aligned_topic IS NOT NULL 
    GROUP BY aligned_topic 
    ORDER BY count DESC
""").fetchall()

for topic, count, percentage in distribution:
    print(f"   • {topic:25s} {count:7,} ({percentage:5.2f}%)")

print(f"\n✅ Database updated: {db_file}")
print(f"\n📊 Tables created:")
print(f"   • conversations_with_topics - Full data with sentiment + topics")
print(f"   • topic_mapping_inspection - Topic mapping for evaluation")
print(f"\n📝 Columns in conversations_with_topics:")
print(f"   • extracted_topic_id - BERTopic assigned topic ID")
print(f"   • extracted_topic_keywords - Raw keywords from BERTopic")
print(f"   • aligned_topic - Aligned predefined topic category")
print(f"   • alignment_confidence - Similarity score (0-1)")
print(f"\n💡 To inspect topic mappings:")
print(f"   duckdb conversations.duckdb -line \"SELECT * FROM topic_mapping_inspection ORDER BY num_conversations DESC LIMIT 10\"")
print()

conn.close()
