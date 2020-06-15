# ATP Interacting Residues

**[Kaggle Link](https://www.kaggle.com/c/iqb213-atp)**

## Syntax Format

```
python3 atp.py
```

## Inputs asked

* **Type of model**
	* SVC: Support Vector Classifier
	* RFC: Random Forest Classifier
* **Window length.** The window length for classifying to the sequence to vectors
* **Balanced Option.** Y/N for balancing the train dataset or not by Balanced Bagging Classifier.
* **Output File.** .csv format is preferred.

## Default values

* Model: SVC
* Window length: 13
* Balanced Option: Y
* Output File: output.csv

## Please Note
Having train.data and test1.txt files in the same folder are mandatory.

## Examples

```
python3 atp.py (kaggle submission 1)

Enter 'd' for default settings for the following
Enter 1 for SVM, Enter 2 for Random Forest Classifier: 1 Enter window size: 17
Enter Y/N to balance training data: Y
Enter output file name: submission1.csv
Model trained!
Output exported in submission1.csv
```
```
python3 atp.py (kaggle submission 2)

Enter 'd' for default settings for the following
Enter 1 for SVM, Enter 2 for Random Forest Classifier: 2 Enter window size: 13
Enter Y/N to balance training data: Y
Enter output file name: submission2.csv
Model trained!
Output exported in submission2.csv
```
```
python3 atp.py

Enter 'd' for default settings for the following
Enter 1 for SVM, Enter 2 for Random Forest Classifier: 1 Enter window size: d
Enter Y/N to balance training data: d
Enter output file name: d
Model trained!
Output exported in output.csv
```