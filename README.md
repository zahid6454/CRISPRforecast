# CRISPRforecast: An effective mechanism to predict high on-target sgRNA activity in CRISPR/Cas9 systems

## Authors: Zahid Hossain, Sajid Ahmed, Shafayat Bin Sabbir, Atikul Islam Sajib, Md Rafsan Jani, Sheikh Adilina , Dewan Md. Farid, Abdollah Dehzangi & Swakkhar Shatabda

## Website Link: https://crisprforecast.pythonanywhere.com/

## Structure:
  - Original Dataset is stored inside the "data" folder.
  - Models and Feature Generation codes are stored inside "codes" folder.
  - All the necessary diagrams are stored inside "graphs and diagrams" folder.

## Employing CRISPRforecast:
  - Install necessary packages from "requirements.txt"
  - In order to generate features, use the dataset that is given inside the "data" folder.
    - Put both Feature_Generation.ipynb and Original_Dataset.csv file in the same folder.
    - Then just execute the Feature_Generation.ipynb file, the output will be a new file titled "Dataset.csv".
    - Dataset.csv holds all the generated features which will be used for CRISPRforecast models.
  - Pull the following files in the same folder:
    - CRISPR_Feature_Extraction_KPCA_Linear.ipynb
    - CRISPR_Feature_Extraction_KPCA_Poly.ipynb
    - CRISPR_Feature_Selection_RFC.ipynb
    - CRISPR_No_Feature_Reduction.ipynb 
    - Dataset.csv
  - Final step, execute every single .ipynb file to get the results of every single CRISPRforecast models.     
