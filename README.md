# CAD Analysis Dashboard  

## Introduction  

The **CAD Analysis Dashboard** is an interactive platform designed to enhance the diagnosis of **Idiopathic Pulmonary Fibrosis (IPF)** and differentiate it from other **Interstitial Lung Diseases (ILDs)**. Built on a Computer-Aided Diagnosis (CAD) system, the dashboard integrates advanced data visualization and radiomic feature analysis, enabling radiologists and researchers to evaluate segmentation performance, explore data, and interpret model predictions.  

## Features  

- **Dataset Overview**:  
  - Display dataset shape, column data types, and descriptive statistics.  
  - Analyze and handle missing values.  

- **Key Performance Indicators (KPIs)**:  
  - Calculate mean, median, and mode for selected numerical features.  
  - Evaluate variability metrics such as standard deviation and interquartile range.  

- **Data Exploration**:  
  - Filter analysis based on categories: IPF, ILD, or both.  
  - Visualize feature distributions, correlations, and outliers.  

- **Segmentation and Classification Results**:  
  - Visualize segmentation masks generated by the U-Net model.  
  - Analyze classification outputs with metrics like confusion matrix, ROC curves, and AUC.  

- **Interactive Visualizations**:  
  - Correlation heatmaps and feature importance graphs.  
  - Dynamic plots for radiomic feature trends and distributions.  

## Technologies Used  

### Backend  
- **Flask**: For hosting API services.  
- **Python**: Core logic for radiomic feature analysis and data handling.  

### Frontend  
- **Dash**: For creating an interactive web-based dashboard.  
- **Plotly**: For advanced and interactive visualizations.  

### Data Processing  
- **PyRadiomics**: To extract radiomic features from segmented CT images.  
- **TensorFlow**: For deep learning-based segmentation (U-Net).  

## Setup  

### Prerequisites  
- Python 3.8+  
- TensorFlow and PyRadiomics installed  

### Installation  

1. **Clone the repository**  
   ```bash  
   git clone https://github.com/yourusername/cad-analysis-dashboard.git  
   cd cad-analysis-dashboard  
   ```  

2. **Set up the Python environment**  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`  
   pip install -r requirements.txt  
   ```  

3. **Run the application**  
   ```bash  
   python app.py  
   ```  

4. **Access the dashboard**  
   Open `http://localhost:8050` in your browser.  

## Usage  

1. **Upload Data**: Import your segmented CT scans and corresponding features for analysis.  
2. **Explore Dataset**: View statistical summaries and manage missing values.  
3. **Analyze Features**: Investigate correlations, variability, and key patterns in radiomic data.  
4. **Evaluate Performance**: Review model performance metrics and interpret classification results.  

## Contributing  

1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature/new-feature`.  
3. Commit changes: `git commit -m 'Add new feature'`.  
4. Push to the branch: `git push origin feature/new-feature`.  
5. Open a pull request.  

## License  

Distributed under the MIT License. See `LICENSE` for details.  
