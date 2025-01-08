# Software Engineering Final Project

This project applies Natural Language Processing (NLP) and Machine Learning (ML) techniques to automate tasks related to bug reports, including classification, prioritization, and team assignment. Below is a detailed explanation of every step in the implementation.

![Architecture Diagram](https://utfs.io/f/Uo47cnkptDxCFtXHwaeL4x7ofgIewABPD2ka53HtvOpZX9CT)

## **1. Project Overview**
This project is part of CSI5137 - Applications of NLP and ML in Software Engineering. It involves:
- **Bug Classification**: Categorizing bug reports by type.
- **Bug Prioritization**: Assigning priority levels to bug reports.
- **Team Assignment**: Assigning bugs to appropriate teams based on description semantics.

## **2. Requirements**
### Libraries and Tools:
- `transformers`, `datasets`, `torch` for BERT-based models.
- `sklearn` for preprocessing and evaluation metrics.
- `nltk` for text tokenization and stopword removal.
- `pandas` for data manipulation.
- `sentence_transformers` for semantic similarity computations.
- `nlpaug` for text augmentation.

Ensure `nltk` resources are downloaded before running the script:
```python
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```

## **3. Implementation Steps**

### **Step 1: Data Preprocessing**
- **Input**: CSV file `jira_flattened_results_clean.csv` containing columns like `fields_summary`, `fields_description`, etc.
- **Processing**:
  1. Missing values in critical columns are replaced with empty strings.
  2. Text is preprocessed using:
     - Lowercasing.
     - Removal of non-alphabetic characters.
     - Tokenization and stopword removal (via NLTK).
  3. Columns `fields_summary` and `fields_description` are combined into `text_combined`.
- **Output**: Processed dataset saved as `preprocessed_jira_data.csv`.

### **Step 2: Bug Classification**
- **Objective**: Classify bugs into categories based on `fields_issuetype_name`.
- **Steps**:
  1. Split the `text_combined` and `fields_issuetype_name` columns into train and test sets.
  2. Tokenize the text using `BertTokenizer` and preprocess labels into numerical format.
  3. Fine-tune a BERT model (`bert-base-uncased`) with:
     - 10 epochs.
     - Batch sizes of 16 (train) and 64 (evaluation).
  4. Evaluate the model using metrics like precision, recall, and F1-score.
  5. Save the model and tokenizer for later use.

### **Step 3: Bug Prioritization**
- **Objective**: Predict priority levels from `fields_priority_name`.
- **Steps**:
  1. Encode priority levels into numeric labels using `LabelEncoder`.
  2. Split data into train, validation, and test sets, ensuring class balance.
  3. Compute class weights to handle imbalanced data.
  4. Fine-tune BERT with:
     - Weighted cross-entropy loss.
     - Custom `Trainer` for handling class weights.
     - 20 epochs for better learning.
  5. Evaluate on the validation and test sets and save the best-performing model.

### **Step 4: Team Assignment**
- **Objective**: Assign bugs to predefined teams based on description semantics.
- **Steps**:
  1. Define team descriptions for `UI`, `Backend`, `DevOps`, and `QA`.
  2. Use `SentenceTransformer` to encode team descriptions and compute semantic embeddings.
  3. Assign bugs to the team with the highest cosine similarity to the bug description.
  4. Employ `nlpaug` for data augmentation to enhance embeddings.

### **Step 5: Unified Task Processing**
- **Objective**: Integrate all tasks into a single function.
- **Steps**:
  1. Load saved models and tokenizers for bug classification and prioritization.
  2. Predict the bug class and priority using the respective models.
  3. Assign the bug to a team using the team assignment function.
  4. Return all results for a given bug description.

### **Step 6: Testing**
- **Example Input**:
  ```
  "allow copy selected lines diff like copy lines diff preview eg use changelog notes commit message"
  ```
- **Output**:
  - **Classification**: Predicted bug type.
  - **Priority**: Predicted priority level.
  - **Assigned Team**: Team responsible for handling the bug.

## **4. Outputs and Saved Models**
1. Preprocessed dataset: `preprocessed_jira_data.csv`.
2. Classification Model: Saved as `bert_classification_model`.
3. Prioritization Model: Saved as `bert_priority_model`.
4. Label encodings and mappings for reuse.

## **5. Key Functions**
1. `preprocess_text(text)`: Cleans and tokenizes text.
2. `predict_bert(texts, model, tokenizer, device)`: Predicts bug types using the classification model.
3. `assign_bug_to_team(text)`: Assigns teams based on semantic similarity.
4. `process_bug(text)`: Combines classification, prioritization, and team assignment.

## **6. Evaluation Metrics**
- **Classification Task**:
  - Precision, Recall, F1-Score for each class.
  - Overall Accuracy.
- **Prioritization Task**:
  - Weighted metrics (Precision, Recall, F1).
  - Test set evaluation metrics.

## **7. How to Run**
1. Ensure required libraries are installed.
2. Place the input CSV file (`jira_flattened_results_clean.csv`) in the same directory.
3. Run the script sequentially to preprocess data, train models, and test outputs.

## **8. Authors**
- Mohammad Bin Yousuf (CU# 101239019)
- Vrishab Prasanth Davey (UO# 300438343)
- Surendar Pala Dana Sekaran (UO#300401916)

## Acknowledgments
This project was developed as part of the CSI5137 - Applications of NLP and ML in Software Engineering course under the guidance of Professor Mehrdad Sabetzadeh.
