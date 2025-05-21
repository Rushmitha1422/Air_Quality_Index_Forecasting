# Air_Quality_Index_Forecasting

## Introduction
Air pollution is a major environmental concern affecting human health and climate conditions. This project predicts **Air Quality Index (AQI)** using **Machine Learning** techniques. The goal is to enhance **prediction accuracy and efficiency** by optimizing model parameters.

## Project Overview
This project implements a **machine learning-based AQI forecasting model** using various datasets containing air pollutant concentrations (**SO2, NO2, PM10, PM2.5, CO, O3**) and environmental factors.

### **Key Features**
- Uses **GA-KELM** to improve prediction accuracy.
- Implements **Logistic Regression, Random Forest, Decision Tree, and SVM** models for comparison.
- Preprocesses datasets by handling missing values and converting data into numerical format.
- Provides **AQI classification**, determining whether air quality is suitable for breathing.
- Enhances training speed and stability compared to traditional models.

## Dataset
The dataset includes multiple air quality parameters collected from **air monitoring stations**:
- **air_quality.csv** - Contains pollutant concentration values.
- **test.csv** - Used for evaluating model predictions.

## Installation & Setup
### **Prerequisites**
- Python **3.7+**
- Libraries: **NumPy, Pandas, Scikit-learn**

### **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/Air_Quality_Index_Forecasting.git
   
2. Navigate to the Project Directory
After cloning the repository, enter the project folder:
    ```bash
     cd Air_Quality_Index_Forecasting
    
3. How to Run the Project
execute the code:
    ```bash
      python air_quality.py
  Or, if using a batch file:  click run.bat file

## System Architecture
The system follows a machine learning pipeline:
- Preprocessing - Cleans missing values and converts data types.
- Training Models - Implements various algorithms for better accuracy.
- Prediction - Predicts AQI levels using trained models.
  
## Results & Model Accuracy
| Model | Accuracy | 
| Logistic Regression | 44.88% | 
| Decision Tree | 64.97% | 
| Random Forest | 74.51% | 
| Support Vector Machine | 66.85% | 


## Future Enhancements
- Implement Deep Learning models (LSTM, CNN) for better accuracy.
- Integrate real-time air quality monitoring using IoT sensors.
- Develop interactive dashboards for AQI visualization.
- Improve dataset quality by adding meteorological factors.











