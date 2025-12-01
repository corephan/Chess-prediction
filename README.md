# Project: Chess Outcomes Prediction

This project aims to build a **Convolutional Neural Network (CNN)** model to analyze the state of a chessboard (represented by the FEN string) at various points during a game and predict the final outcome: **White Win (1-0)**, **Black Win (0-1)**, or **Draw (1/2-1/2)**.

---

## Project Goals

1.  **Classification Model:** Develop a CNN capable of processing the $8 \times 8$ board matrix as input.
2.  **3-Class Prediction:** Successfully classify the game outcome into three distinct labels.
3.  **Reproducibility:** Ensure the code is easily runnable on Google Colab and local Python virtual environments.

---

## Technology Stack and Libraries

* **Language:** Python 3.x
* **ML Framework:** TensorFlow / Keras (or PyTorch)
* **Chess Data Handling:** `python-chess`, `numpy`, `pandas`
* **Environment:** Google Colab (recommended) and VS Code/Jupyter Notebook (local).

---

## Dataset Overview

### 1. Raw Data Source
* **Source:** Lichess Database (Raw PGN files).
* **Size:** [Insert Total Number of Games / Total PGN file size here].

### 2. Data Preprocessing
* **FEN Extraction:** Data is extracted from PGN files, sampling the board state (FEN) every **[Example: 10]** moves to capture various game stages.
* **Encoding:** The FEN string is converted into a **multi-channel numerical Tensor** (e.g., $8 \times 8 \times 14$) suitable as input for the CNN model.

---

## Model Architecture (CNN)

The model employs a standard Convolutional Neural Network (CNN) architecture designed to recognize spatial patterns and relationships between pieces on the $8 \times 8$ board.

* **Input:** Tensor ($8 \times 8 \times K$ channels, where $K$ is the number of encoding layers).
* **Key Layers:**
    * [Number] **Convolutional (Conv2D)** layers (using Kernel size and ReLU Activation).
    * **Pooling/Flattening** layers.
    * [Number] **Fully Connected (Dense)** layers.
* **Output Layer:** 3 nodes with the **Softmax** activation function (representing White Win / Black Win / Draw probabilities).
* **Parameters:**
    * Loss Function: Categorical Cross-Entropy.
    * Optimizer: Adam.

---

## Setup and Running Instructions

### 1. Virtual Environment Setup
```bash
python -m venv venv
# Activate the environment
# Windows: .\venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

2. Install Dependencies
pip install tensorflow python-chess numpy pandas matplotlib

3. Execution
Run data_extraction.ipynb or data_extraction.py to process the raw PGN data.

Run train_model.ipynb or train_model.py to train the prediction model.

Role,Member,Primary Responsibilities
Team Lead / Data Lead,Member 1,"Dataset Acquisition, Report Aggregation, Progress Management."
ML Engineer,Member 2,"FEN-to-Tensor Preprocessing, Baseline Model Training."
System / Code Lead,Member 3,"Code Structure, Environment Management (Colab/Drive), Demo Preparation."
Algorithm Specialist,Member 4,"CNN Architecture Design, Slide Preparation."
Optimization Specialist,Member 5,"Hyper-parameter Tuning, Performance Analysis."

Project Submission
All project files and documentation will be shared via Google Drive based on the required folder structure:

Presentation/: Presentation slides (15 minutes).

Report/: Complete technical report (docx/pptx).

Demo/: Simple demonstration code and sample dataset.

System/: Complete Code, Saved Model, Final Dataset.
