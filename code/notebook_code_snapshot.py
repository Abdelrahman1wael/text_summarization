
# ================= CELL 1 =================

import kagglehub

# Download latest version
path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")

print("Path to dataset files:", path)



# ================= CELL 2 =================

get_ipython().system('pip install -q transformers rouge-score pandas torch')



# ================= CELL 3 =================

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # swap for bart-large-cnn / pegasus-cnn_dailymail / t5-small
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()



# ================= CELL 4 =================

# Adjust path if your unzipped folder is named differently
df = pd.read_csv("test.csv")  # columns typically: id, article, highlights
print(df.shape)
print(df.columns.tolist())
print(df.head(2))

# Small demo sample (increase after verifying the pipeline)
df_sample = df.head(20).copy()  # or df.sample(20, random_state=42)



# ================= CELL 5 =================

get_ipython().system('pip -q install transformers datasets evaluate rouge_score sentencepiece accelerate kagglehub pandas tqdm')



# ================= CELL 6 =================

import os
import glob
import re
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

pd.set_option("display.max_colwidth", 120)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Encoder-decoder summarization model
MODEL_NAME = "facebook/bart-large-cnn"

# Faster option for low GPU/CPU:
# MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Other possible models:
# MODEL_NAME = "google/pegasus-cnn_dailymail"
# MODEL_NAME = "t5-small"

N_EVAL = 50          # Increase to 200/500 for stronger evaluation
BATCH_SIZE = 2       # Use 4 or 8 if you have enough GPU memory



# ================= CELL 7 =================

DATASET_SLUG = "thedevastator/cnn-dailymail-summarization-dataset"

csv_files = []
data_path = None

try:
    import kagglehub
    
    data_path = kagglehub.dataset_download(DATASET_SLUG)
    print("Dataset downloaded to:", data_path)
    
    csv_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)
    print("CSV files found:", len(csv_files))
    
    for f in csv_files[:10]:
        print(f)

except Exception as e:
    print("Kaggle download failed.")
    print("Error:", e)
    print("Fallback will use Hugging Face CNN/DailyMail dataset.")



# ================= CELL 8 =================

import kagglehub

# Download latest version
path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")

print("Path to dataset files:", path)



# ================= CELL 9 =================

get_ipython().system('pip -q install transformers evaluate rouge_score sentencepiece accelerate pandas tqdm')



# ================= CELL 10 =================

import os
import glob

csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

print("CSV files found:")
for file in csv_files:
    print(file)



# ================= CELL 11 =================

import pandas as pd

# Pick train/test/validation files automatically
train_file = None
test_file = None
val_file = None

for file in csv_files:
    name = os.path.basename(file).lower()
    
    if "train" in name:
        train_file = file
    elif "test" in name:
        test_file = file
    elif "validation" in name or "valid" in name or "val" in name:
        val_file = file

print("Train file:", train_file)
print("Validation file:", val_file)
print("Test file:", test_file)

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

if val_file is not None:
    val_df = pd.read_csv(val_file)
else:
    val_df = None

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

train_df.head()



# ================= CELL 12 =================

import re

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_dataframe(df):
    df = df.copy()
    
    print("Original columns:", df.columns.tolist())
    
    # Common CNN/DailyMail column names
    if "article" in df.columns and "highlights" in df.columns:
        df = df[["article", "highlights"]]
        df.columns = ["article", "summary"]
        
    elif "article" in df.columns and "summary" in df.columns:
        df = df[["article", "summary"]]
        
    elif "text" in df.columns and "summary" in df.columns:
        df = df[["text", "summary"]]
        df.columns = ["article", "summary"]
        
    else:
        raise ValueError("Could not identify article and summary columns.")
    
    df["article"] = df["article"].apply(clean_text)
    df["summary"] = df["summary"].apply(clean_text)
    
    df = df.dropna()
    df = df[df["article"].str.len() > 100]
    df = df[df["summary"].str.len() > 10]
    
    return df.reset_index(drop=True)


train_df = prepare_dataframe(train_df)
test_df = prepare_dataframe(test_df)

if val_df is not None:
    val_df = prepare_dataframe(val_df)

print("Clean train shape:", train_df.shape)
print("Clean test shape:", test_df.shape)

train_df.head()



# ================= CELL 13 =================

import torch
import numpy as np
import random
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)



# ================= CELL 14 =================

MODEL_NAME = "facebook/bart-large-cnn"

# If your system is slow or GPU memory is low, use this smaller model:
# MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

model.eval()

print("Model loaded:", MODEL_NAME)



# ================= CELL 15 =================

MAX_INPUT_TOKENS = 1024
MAX_SUMMARY_TOKENS = 128
MIN_SUMMARY_TOKENS = 30

sample_article = test_df.loc[0, "article"]

tokens_without_truncation = tokenizer(
    sample_article,
    truncation=False,
    add_special_tokens=True
)["input_ids"]

tokens_with_truncation = tokenizer(
    sample_article,
    max_length=MAX_INPUT_TOKENS,
    truncation=True,
    add_special_tokens=True
)["input_ids"]

print("Original token length:", len(tokens_without_truncation))
print("After truncation:", len(tokens_with_truncation))



# ================= CELL 16 =================

def generate_summaries(articles, batch_size=2):
    generated_summaries = []
    
    for i in tqdm(range(0, len(articles), batch_size)):
        batch_articles = articles[i:i + batch_size]
        
        inputs = tokenizer(
            batch_articles,
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=MAX_SUMMARY_TOKENS,
                min_length=MIN_SUMMARY_TOKENS,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        batch_summaries = tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        generated_summaries.extend(batch_summaries)
    
    return generated_summaries



# ================= CELL 17 =================

idx = 0

article = test_df.loc[idx, "article"]
reference_summary = test_df.loc[idx, "summary"]

generated_summary = generate_summaries([article], batch_size=1)[0]

print("ARTICLE:")
print(article[:2000])

print("\n" + "="*80 + "\n")

print("REFERENCE SUMMARY:")
print(reference_summary)

print("\n" + "="*80 + "\n")

print("GENERATED SUMMARY:")
print(generated_summary)



# ================= CELL 18 =================

rouge = evaluate.load("rouge")

N_EVAL = 50

eval_df = test_df.sample(
    n=min(N_EVAL, len(test_df)),
    random_state=SEED
).reset_index(drop=True)

predictions = generate_summaries(
    eval_df["article"].tolist(),
    batch_size=2
)

references = eval_df["summary"].tolist()

rouge_scores = rouge.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True
)

rouge_scores



# ================= CELL 19 =================

rouge_df = pd.DataFrame({
    "Metric": list(rouge_scores.keys()),
    "Score": [round(value * 100, 2) for value in rouge_scores.values()]
})

rouge_df



# ================= CELL 20 =================

results_df = eval_df.copy()
results_df["generated_summary"] = predictions

results_df.to_csv("cnn_dailymail_generated_summaries.csv", index=False)

results_df.head()



# ================= CELL 21 =================

for i in range(3):
    print("ARTICLE:")
    print(results_df.loc[i, "article"][:1000])
    
    print("\nREFERENCE SUMMARY:")
    print(results_df.loc[i, "summary"])
    
    print("\nGENERATED SUMMARY:")
    print(results_df.loc[i, "generated_summary"])
    
    print("\n" + "="*100 + "\n")



# ================= CELL 22 =================

# ===================== EXPORT WHOLE PROJECT AS ZIP =====================

import os
import shutil
import json
import glob
import datetime
from pathlib import Path

# Change to True only if you also want Hugging Face downloaded model weights.
# Warning: this can make the zip very large.
INCLUDE_MODEL_CACHE = False

PROJECT_NAME = "text_summarization_project_export"

# Export location
base_dir = Path("/content") if Path("/content").exists() else Path.cwd()
export_dir = base_dir / PROJECT_NAME
zip_path = base_dir / f"{PROJECT_NAME}.zip"

# Remove old export if it exists
if export_dir.exists():
    shutil.rmtree(export_dir)

if zip_path.exists():
    zip_path.unlink()

# Create folders
(export_dir / "workspace").mkdir(parents=True, exist_ok=True)
(export_dir / "data").mkdir(parents=True, exist_ok=True)
(export_dir / "outputs").mkdir(parents=True, exist_ok=True)
(export_dir / "code").mkdir(parents=True, exist_ok=True)
(export_dir / "metadata").mkdir(parents=True, exist_ok=True)

print("Export folder created:", export_dir)

# ---------------------------------------------------------------------
# 1. Save notebook/code snapshot from executed cells
# ---------------------------------------------------------------------
try:
    ip = get_ipython()
    cells = ip.user_ns.get("In", [])

    code_snapshot = []
    for i, cell in enumerate(cells):
        if str(cell).strip():
            code_snapshot.append(f"\n# ================= CELL {i} =================\n")
            code_snapshot.append(str(cell))
            code_snapshot.append("\n")

    with open(export_dir / "code" / "notebook_code_snapshot.py", "w", encoding="utf-8") as f:
        f.write("\n".join(code_snapshot))

    print("Saved code snapshot.")
except Exception as e:
    print("Could not save code snapshot:", e)

# ---------------------------------------------------------------------
# 2. Save important variables and results if they exist
# ---------------------------------------------------------------------
metadata = {
    "export_time": str(datetime.datetime.now()),
    "project": "Text Summarization using Pre-trained Encoder-Decoder Models",
    "dataset": "CNN/DailyMail News Summarization",
}

for var_name in [
    "MODEL_NAME",
    "MAX_INPUT_TOKENS",
    "MAX_SUMMARY_TOKENS",
    "MIN_SUMMARY_TOKENS",
    "N_EVAL",
    "BATCH_SIZE",
    "SEED"
]:
    if var_name in globals():
        try:
            metadata[var_name] = str(globals()[var_name])
        except:
            pass

# Save ROUGE scores if available
if "rouge_scores" in globals():
    try:
        rouge_scores_clean = {k: float(v) for k, v in rouge_scores.items()}
        metadata["rouge_scores"] = rouge_scores_clean

        with open(export_dir / "outputs" / "rouge_scores.json", "w", encoding="utf-8") as f:
            json.dump(rouge_scores_clean, f, indent=4)

        print("Saved ROUGE scores.")
    except Exception as e:
        print("Could not save ROUGE scores:", e)

# Save results dataframe if available
if "results_df" in globals():
    try:
        results_df.to_csv(export_dir / "outputs" / "generated_summaries.csv", index=False)
        print("Saved generated summaries.")
    except Exception as e:
        print("Could not save results_df:", e)

# Save sample cleaned dataframes if available
for df_name in ["train_df", "val_df", "test_df"]:
    if df_name in globals():
        try:
            df = globals()[df_name]
            df.head(1000).to_csv(export_dir / "outputs" / f"{df_name}_sample_1000.csv", index=False)
            metadata[f"{df_name}_shape"] = str(df.shape)
        except Exception as e:
            print(f"Could not save {df_name} sample:", e)

# Save metadata
with open(export_dir / "metadata" / "project_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

# ---------------------------------------------------------------------
# 3. Save requirements file
# ---------------------------------------------------------------------
requirements = """
transformers
evaluate
rouge_score
sentencepiece
accelerate
pandas
torch
numpy
tqdm
kagglehub
"""

with open(export_dir / "requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements.strip())

print("Saved requirements.txt.")

# ---------------------------------------------------------------------
# 4. Copy Kaggle dataset files
# ---------------------------------------------------------------------
dataset_path = None

if "path" in globals() and os.path.exists(str(path)):
    dataset_path = Path(path)

elif "data_path" in globals() and os.path.exists(str(data_path)):
    dataset_path = Path(data_path)

if dataset_path is not None:
    try:
        shutil.copytree(
            dataset_path,
            export_dir / "data" / "cnn_dailymail_dataset",
            dirs_exist_ok=True
        )
        metadata["dataset_path"] = str(dataset_path)
        print("Copied dataset from:", dataset_path)
    except Exception as e:
        print("Could not copy dataset folder:", e)

elif "csv_files" in globals():
    try:
        for file in csv_files:
            shutil.copy(file, export_dir / "data" / os.path.basename(file))
        print("Copied CSV dataset files.")
    except Exception as e:
        print("Could not copy CSV files:", e)

else:
    print("No dataset path variable found. Dataset was not copied.")

# ---------------------------------------------------------------------
# 5. Copy current workspace project files
# ---------------------------------------------------------------------
def ignore_files(dir, files):
    ignore_list = []
    for file in files:
        if file in [PROJECT_NAME, f"{PROJECT_NAME}.zip", "sample_data", "__pycache__", ".ipynb_checkpoints"]:
            ignore_list.append(file)
        if file.endswith(".zip"):
            ignore_list.append(file)
    return ignore_list

try:
    shutil.copytree(
        Path.cwd(),
        export_dir / "workspace",
        dirs_exist_ok=True,
        ignore=ignore_files
    )
    print("Copied current workspace files.")
except Exception as e:
    print("Could not copy full workspace:", e)

# ---------------------------------------------------------------------
# 6. Optionally copy Hugging Face model cache
# ---------------------------------------------------------------------
if INCLUDE_MODEL_CACHE:
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        try:
            shutil.copytree(
                hf_cache,
                export_dir / "models" / "huggingface_cache",
                dirs_exist_ok=True
            )
            print("Copied Hugging Face model cache.")
        except Exception as e:
            print("Could not copy Hugging Face cache:", e)
    else:
        print("No Hugging Face cache found.")

# ---------------------------------------------------------------------
# 7. Create ZIP file
# ---------------------------------------------------------------------
archive_path = shutil.make_archive(
    base_name=str(zip_path).replace(".zip", ""),
    format="zip",
    root_dir=export_dir
)

print("\nZIP export completed successfully!")
print("ZIP file path:", archive_path)

# ---------------------------------------------------------------------
# 8. Download ZIP automatically in Google Colab
# ---------------------------------------------------------------------
try:
    from google.colab import files
    files.download(archive_path)
except Exception:
    print("If download did not start automatically, download this file manually:")
    print(archive_path)



# ================= CELL 23 =================

# ===================== EXPORT WHOLE PROJECT AS ZIP =====================

import os
import shutil
import json
import glob
import datetime
from pathlib import Path

# Change to True only if you also want Hugging Face downloaded model weights.
# Warning: this can make the zip very large.
INCLUDE_MODEL_CACHE = False

PROJECT_NAME = "text_summarization_project_export"

# Export location
base_dir = Path("/content") if Path("/content").exists() else Path.cwd()
export_dir = base_dir / PROJECT_NAME
zip_path = base_dir / f"{PROJECT_NAME}.zip"

# Remove old export if it exists
if export_dir.exists():
    shutil.rmtree(export_dir)

if zip_path.exists():
    zip_path.unlink()

# Create folders
(export_dir / "workspace").mkdir(parents=True, exist_ok=True)
(export_dir / "data").mkdir(parents=True, exist_ok=True)       # kept for outputs/manual files
(export_dir / "outputs").mkdir(parents=True, exist_ok=True)
(export_dir / "code").mkdir(parents=True, exist_ok=True)
(export_dir / "metadata").mkdir(parents=True, exist_ok=True)

print("Export folder created:", export_dir)

# ---------------------------------------------------------------------
# 1. Save notebook/code snapshot from executed cells
# ---------------------------------------------------------------------
try:
    ip = get_ipython()
    cells = ip.user_ns.get("In", [])

    code_snapshot = []
    for i, cell in enumerate(cells):
        if str(cell).strip():
            code_snapshot.append(f"\n# ================= CELL {i} =================\n")
            code_snapshot.append(str(cell))
            code_snapshot.append("\n")

    with open(export_dir / "code" / "notebook_code_snapshot.py", "w", encoding="utf-8") as f:
        f.write("\n".join(code_snapshot))

    print("Saved code snapshot.")
except Exception as e:
    print("Could not save code snapshot:", e)

# ---------------------------------------------------------------------
# 2. Save important variables and results if they exist
# ---------------------------------------------------------------------
metadata = {
    "export_time": str(datetime.datetime.now()),
    "project": "Text Summarization using Pre-trained Encoder-Decoder Models",
    "dataset": "CNN/DailyMail News Summarization",
}

for var_name in [
    "MODEL_NAME",
    "MAX_INPUT_TOKENS",
    "MAX_SUMMARY_TOKENS",
    "MIN_SUMMARY_TOKENS",
    "N_EVAL",
    "BATCH_SIZE",
    "SEED"
]:
    if var_name in globals():
        try:
            metadata[var_name] = str(globals()[var_name])
        except:
            pass

# Save ROUGE scores if available
if "rouge_scores" in globals():
    try:
        rouge_scores_clean = {k: float(v) for k, v in rouge_scores.items()}
        metadata["rouge_scores"] = rouge_scores_clean

        with open(export_dir / "outputs" / "rouge_scores.json", "w", encoding="utf-8") as f:
            json.dump(rouge_scores_clean, f, indent=4)

        print("Saved ROUGE scores.")
    except Exception as e:
        print("Could not save ROUGE scores:", e)

# Save results dataframe if available
if "results_df" in globals():
    try:
        results_df.to_csv(export_dir / "outputs" / "generated_summaries.csv", index=False)
        print("Saved generated summaries.")
    except Exception as e:
        print("Could not save results_df:", e)

# Save sample cleaned dataframes if available
for df_name in ["train_df", "val_df", "test_df"]:
    if df_name in globals():
        try:
            df = globals()[df_name]
            df.head(1000).to_csv(export_dir / "outputs" / f"{df_name}_sample_1000.csv", index=False)
            metadata[f"{df_name}_shape"] = str(df.shape)
        except Exception as e:
            print(f"Could not save {df_name} sample:", e)

# Save metadata
with open(export_dir / "metadata" / "project_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

# ---------------------------------------------------------------------
# 3. Save requirements file
# ---------------------------------------------------------------------
requirements = """
transformers
evaluate
rouge_score
sentencepiece
accelerate
pandas
torch
numpy
tqdm
kagglehub
"""

with open(export_dir / "requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements.strip())

print("Saved requirements.txt.")

# ---------------------------------------------------------------------
# 4. [REMOVED] Copy Kaggle dataset files
# ---------------------------------------------------------------------
# The raw dataset folder is intentionally excluded to keep the zip small.
# If you need it, uncomment/re-add the dataset copying logic here.

# ---------------------------------------------------------------------
# 5. Copy current workspace project files
# ---------------------------------------------------------------------
def ignore_files(dir, files):
    ignore_list = []
    for file in files:
        if file in [PROJECT_NAME, f"{PROJECT_NAME}.zip", "sample_data", "__pycache__", ".ipynb_checkpoints"]:
            ignore_list.append(file)
        if file.endswith(".zip"):
            ignore_list.append(file)
    return ignore_list

try:
    shutil.copytree(
        Path.cwd(),
        export_dir / "workspace",
        dirs_exist_ok=True,
        ignore=ignore_files
    )
    print("Copied current workspace files.")
except Exception as e:
    print("Could not copy full workspace:", e)

# ---------------------------------------------------------------------
# 6. Optionally copy Hugging Face model cache
# ---------------------------------------------------------------------
if INCLUDE_MODEL_CACHE:
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        try:
            shutil.copytree(
                hf_cache,
                export_dir / "models" / "huggingface_cache",
                dirs_exist_ok=True
            )
            print("Copied Hugging Face model cache.")
        except Exception as e:
            print("Could not copy Hugging Face cache:", e)
    else:
        print("No Hugging Face cache found.")

# ---------------------------------------------------------------------
# 7. Create ZIP file
# ---------------------------------------------------------------------
archive_path = shutil.make_archive(
    base_name=str(zip_path).replace(".zip", ""),
    format="zip",
    root_dir=export_dir
)

print("\nZIP export completed successfully!")
print("ZIP file path:", archive_path)

# ---------------------------------------------------------------------
# 8. Download ZIP automatically in Google Colab
# ---------------------------------------------------------------------
try:
    from google.colab import files
    files.download(archive_path)
except Exception:
    print("If download did not start automatically, download this file manually:")
    print(archive_path)

