# Timestamp Segmentation and Repair

This project provides implementations for timestamp segmentation and repair methods, including SegRIR-exact and SegRIR-appr, along with baseline approaches.

## 📁 Project Structure

- `code/`: Contains all implementation code for segmentation and repair.
  - `s-main.py`: Main entry point for running the SegRIR method.
  - `exact.py`: Implementation of the **SegRIR-exact** exact repair method.
  - `appr.py`: Implementation of the **SegRIR-appr** approximation repair method.
  - `metrics.py`: Defines the evaluation metrics used to assess repair quality.
  - `screen.py`: Implementation of the **SCREEN** baseline method.
  - `var.py`: Implementation of the **VAR** baseline method.
- `data/`: Contains datasets used for evaluation.
  - Each `.csv` file includes two columns:
    - Ground truth timestamps
    - Dirty timestamps

- For the **RIR method**, please see the official repository:  
  [https://github.com/fangfcg/regular-interval-repair](https://github.com/fangfcg/regular-interval-repair)

## 🚀 Running the Example

> ⚠️ Recommended Python version: 3.10 
```bash
pip install -r requirements.txt
cd code/
python s-main.py
