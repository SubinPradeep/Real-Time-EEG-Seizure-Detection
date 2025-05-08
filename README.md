# Real-Time EEG Seizure Detection

This repository contains code for reproducing the work on real-time EEG-based seizure detection, based on the paper "Real-Time Seizure Detection using EEG: A Comprehensive Comparison of Recent Approaches under a Realistic Setting" by Lee et al. (CHIL 2022).

## Report
[https://drive.google.com/file/d/15bQVRILmg7UyBtu6aMdFtRYtPrIbnsgm/view](https://drive.google.com/file/d/15bQVRILmg7UyBtu6aMdFtRYtPrIbnsgm/view)

## Orignal Paper
```bibtex
@inproceedings{lee2022real,
  title={Real-Time Seizure Detection using EEG: A Comprehensive Comparison of Recent Approaches under a Realistic Setting},
  author={Lee, Kwanhyung and Jeong, Hyewon and Kim, Seyun and Yang, Donghwa and Kang, Hoon-Chul and Choi, Edward},
  booktitle={Conference on Health, Inference, and Learning},
  pages={311--337},
  year={2022},
  organization={PMLR}
}
```

## Prerequisites

Before you begin, ensure you have the following installed:
* Python (the version used for development, e.g., Python 3.8+)
* pip (Python package installer)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SubinPradeep/Real-Time-EEG-Seizure-Detection.git
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

4. **Focal Loss (Paper Extension)**
    To run the focal loss extension of the implementation run:
    ```
    python focal_loss_model.py
    ```

4. **Results**
    Our model achieves the following performance metrics:
    - acc: 0.9996
    - rec: 0.9954
    - roc_auc: 0.9999