# Real-Time EEG Seizure Detection

This project implements a real-time EEG seizure detection system as described in the reproduced paper. It utilizes Electroencephalogram (EEG) data to identify seizure activity.

## Description

The primary component of this project is a Jupyter Notebook (`Real_Time_EEG_Seizure_Detection.ipynb`) that walks through the data loading, preprocessing, model training (or loading of a pre-trained model), and real-time simulation or testing of seizure detection.

## Prerequisites

Before you begin, ensure you have the following installed:
* Python (the version used for development, e.g., Python 3.8+)
* pip (Python package installer)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SubinPradeep/Real-Time-EEG-Seizure-Detection.git](https://github.com/SubinPradeep/Real-Time-EEG-Seizure-Detection.git)
    cd Real-Time-EEG-Seizure-Detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    * On Windows:
        ```bash
        venv\Scripts\activate
        ```
    * On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    The required Python libraries are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Implementation Workflow

1. **Data**
    This reproduced implementation of the paper generates data synthetically. To generate your data simply run:
    ```
    python data.py
    ```

2. **Data Preprocessing**
    Next to preprocess our data run:
    ```
    python preprocess.py
    ```

3. **Model Training and Evaluation**
    Finally, to run the model on our generated data run:
    ```
    python model.py
    ```
4. **Results**
    Our model achieves the following performance metrics:
    - acc: 0.9996
    - rec: 0.9954
    - roc_auc: 0.9999