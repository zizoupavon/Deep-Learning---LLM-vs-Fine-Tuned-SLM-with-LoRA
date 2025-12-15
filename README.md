# Comparing Two AI Pathways for Intent Classification

## 📌 Project Overview

Customer support teams receive thousands of short messages every day, such as:

- “Where’s my order?”
- “Cancel my subscription”
- “Update my payment method”

Automatically identifying the **intent** behind these messages enables faster routing, lower costs, and better customer experience.

This project evaluates **two competing AI approaches** for intent classification using the same dataset, splits, and evaluation metrics:

1. **Zero- / Few-Shot classification using Large Language Models (LLMs)**
2. **Fine-tuned transformer-based classifiers**

The goal is to provide a **fair, data-backed technical and business comparison** to inform real-world deployment decisions.

---

## 🎯 Project Objectives

- Build and evaluate two AI pathways for intent classification
- Compare performance using **Accuracy**, **Macro-F1**, **Time**, and **Cost**
- Analyze trade-offs in scalability, latency, and operational complexity
- Deliver a **clear business recommendation** for different company sizes and volumes

---

## 📂 Dataset

**Bitext Customer Support Dataset**  
🔗 https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset

Each record contains:
- `instruction`: Customer message
- `intent`: Labeled intent (e.g., `track_order`, `refund_request`, `update_account`)

### Data Splits
- Train / Validation / Test split: **80 / 10 / 10**
- The same splits and label sets are used across all experiments

---

## 🧹 Data Preprocessing

Before training or evaluation, placeholder tokens are replaced with realistic synthetic values to simulate real customer messages and improve generalization.

### Placeholder Replacement Examples

| Placeholder | Example Replacement |
|------------|---------------------|
| `{{order_number}}` | `ORD-2025-12345` |
| `{{customer_name}}` | `Maria Garcia` |
| `{{email}}` | `alex.smith@example.com` |
| `{{product_name}}` | `Wireless Mouse` |
| `{{date}}` | `05/09/2025` |

### Preprocessing Steps
1. Load the CSV dataset into a DataFrame
2. Detect placeholders using regex: `r"\{\{([^}]+)\}\}"`
3. Replace each placeholder with a type-specific synthetic value
4. Sample and manually verify 10–20 processed rows
5. Save the cleaned dataset for downstream experiments

> 📄 The preprocessing logic and design decisions are documented in the technical report.

---

## 🧠 Modeling Approaches

---

## Approach 1: Large Language Model (LLM) Inference

### Description
This approach uses a **large general-purpose language model** to classify customer intent via prompting, without task-specific fine-tuning.

### Model Used
- **GPT-4.1 Mini, GPT-4, Claude, Gemini **** (via API)

### Prompting Strategy
- **Zero-shot prompting** was used as the primary strategy
- A structured list of all possible intents was included in the prompt
- Few-shot prompting was evaluated conceptually but not used in the final reported results to control API cost and latency

### Evaluation Setup
- ~270 test samples (≈10 per intent class)
- Metrics:
  - Accuracy
  - Macro-F1
- Additional tracking:
  - Estimated inference time per 1,000 samples
  - Approximate API cost

### Key Characteristics
- No training required
- Fast to prototype
- Higher per-request cost
- Latency depends on external API calls

---

## Approach 2: Fine-Tuned Small Language Model (SLM)

### Description
This approach fine-tunes a **small, open-source transformer model** specifically for intent classification using labeled customer support data.

### Model Used
- **DistilRoBERTa**
- Selected for its strong performance-to-size tradeoff and suitability for production deployment

### Fine-Tuning Strategy
Two fine-tuning strategies were considered:

1. **Full Fine-Tuning** (conceptual baseline)
2. **LoRA Fine-Tuning** (implemented)

➡️ **LoRA fine-tuning was used in the final experiments** to reduce training cost and number of trainable parameters while maintaining performance.

### Training Setup
- Framework: HuggingFace Transformers + PEFT (LoRA)
- GPU: Kaggle T4
- Training parameters controlled via `configs/finetune.yaml`
- Model trained on the same dataset split used in Approach 1

### Evaluation Metrics
- Accuracy
- Macro-F1
- Training time
- Estimated compute cost

### Key Characteristics
- Requires upfront training
- Very low inference latency
- Low marginal cost at scale
- Fully deployable on private infrastructure

---

## Unified Evaluation Framework

| Metric | Description |
|------|------------|
| Accuracy | Percentage of correctly predicted intents |
| Macro-F1 | Average F1-score across all intent classes |
| Time | Inference or training time per 1,000 samples |
| Cost | API cost (LLM) or GPU compute estimate |
| Model Size | Number of trainable parameters (reference) |

All comparisons are based on **identical intent labels and data splits**.

---

## Repository Structure
