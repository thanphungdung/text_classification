from classifier import classify_text_and_files
import pandas as pd

# === Configuration ===
# Simulated user input
typed_input = "The market is showing signs of recovery after a tough quarter."

# Paths to test files (you can change these to your real local test files)
file_paths = [
    "test_data/sample.csv",
    "test_data/news_article.pdf",
    "test_data/comments.txt"
]

# Classification task to test (choose from: "Sentiment Analysis", "Spam Detection", "Topic Classification")
task = "Topic Classification"

# === Run classification ===
try:
    result_df = classify_text_and_files(typed_input, file_paths, task)
    
    # Show result
    pd.set_option("display.max_colwidth", None)
    print("\n=== CLASSIFICATION RESULTS ===\n")
    print(result_df)
    
    # Optionally save to CSV
    result_df.to_csv("classification_results.csv", index=False)
    print("\nResults saved to classification_results.csv")

except Exception as e:
    print(f"Error during classification: {e}")
