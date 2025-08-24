# MNIST Classification in R

## Project Overview
This project implements a handwritten digit classification model using the MNIST dataset in R. The goal is to build a machine learning model that can accurately classify digits from 0 to 9. The project focuses on:

- Data preprocessing
- Model training and evaluation
- Calculation of evaluation metrics such as Precision, Recall, and F1-score

All code is contained in the `MNIST.R` script.

---

## File Structure
minst/

├─ MNIST.R # Main R script

├─ README.md # Project description (this file)

└─ .gitignore # Optional, for ignoring temporary files

---

## Requirements
You need the following R packages installed:

```r
install.packages(c("keras", "tensorflow", "dplyr", "yardstick"))
```
Ensure that Keras and TensorFlow are correctly installed. You can follow the Keras for R installation guide
 if needed.

## Usage

Open the MNIST.R script in R or RStudio.

Run the script:

```r
source("MNIST.R")
```

The script will:

 - Load the MNIST dataset

 - Preprocess the images and labels

 - Train a classification model

 - Evaluate the model on the test set

 - Calculate per-class metrics (Precision, Recall, F1-score)

### Evaluation Metrics

  ## The script computes:

    - Test loss and accuracy
       Example output:
         Test loss: 0.0284
         Test accuracy: 0.9915


   ## Per-class metrics (Precision, Recall, F1-score)

      -Example table:

| Class | Precision | Recall | F1    |
|-------|-----------|--------|-------|
| 0     | 0.993     | 0.997  | 0.995 |
| 1     | 0.995     | 0.997  | 0.996 |
| 2     | 0.987     | 0.996  | 0.991 |
| …     | …         | …      | …     |
| 9     | 0.996     | 0.974  | 0.985 |




##  Summary

The trained model achieves high accuracy on the MNIST test set (~99%).
Per-class evaluation shows strong performance across most digits, though some digits may have slightly lower F1-scores due to class imbalance. Overall, the results demonstrate the effectiveness of the chosen model and preprocessing pipeline.



## Notes

The MNIST dataset is automatically downloaded within the script using Keras.

No additional images or datasets are required to run the code.

The script uses R’s yardstick package to compute metrics without relying on Python.
