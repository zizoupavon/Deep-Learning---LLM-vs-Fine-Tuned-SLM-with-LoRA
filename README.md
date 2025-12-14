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

### Approach 1: Zero- / Few-Shot Classification with LLMs

- Models tested: GPT-4, Claude, Gemini (via API)
- Uses a list of intents and optional few-shot examples
- Evaluated on **~270 samples** (≈10 per intent class) to control API cost
- Tracks:
  - Accuracy
  - Macro-F1
  - Inference time per 1,000 samples (estimated)
  - Approximate API cost (USD)

**Output Metrics**
