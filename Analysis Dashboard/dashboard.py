import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from modules.eda import dataset_overview,handle_missing_values, correlation_matrix, top_correlated_features, kpi_and_visualization, remove_outliers_isolation_forest
from modules.eda import display_textual_outlier_comparison, visualize_top_features_box_plots, log_transform, normalize_data, perform_normality_tests
from modules.eda import plot_histograms_before_after, plot_qq_before_after_with_plotly,apply_transformations, perform_t_test
from modules.eda import perform_pca, plot_pca_2d, pca_analysis, train_and_compare_classification_models, show_textual_report, show_visualizations

# Database connection
conn = sqlite3.connect('radiomics_data.db')

@st.cache_data(ttl=7200)  # Disable caching for debugging
def load_data():
    query = "SELECT * FROM radiomic_features"
    return pd.read_sql(query, conn)

# Page configuration
st.set_page_config(page_title="COVID-19 Radiomics Dashboard", layout="wide")

# Custom CSS for blue-shaded dark-themed styling
blue_theme_css = """
    <style>
        .main-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #FFFFFF;
            text-align: center;
            padding: 10px;
        }
        .kpi-card {
            background-color: #2E3B55;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #FFFFFF;
        }
        .sidebar .sidebar-content {
            background-color: #2E3B55;
            color: #FFFFFF;
        }
        .dark-theme {
            background-color: #2E3B55;
            color: #FFFFFF;
        }
        .output-container {
            margin-top: 30px;
        }
        .main-container {
            background-color: #243B53;
        }
    </style>
"""

# Apply blue shade theme CSS
st.markdown(blue_theme_css, unsafe_allow_html=True)

st.sidebar.title("COVID-19 Radiomics Dashboard")
page_selection = st.sidebar.radio("Navigate to:", ["Textual Analysis", "Visual Comparisons"])
# Normalization Methods in the sidebar
normalization_method = st.sidebar.radio(
    "Select Normalization Method",
    ('Min-Max', 'Z-score', 'Max-Abs')
)


 # Load and filter data based on selection
data_df = load_data()
if "Target" in data_df.columns:
    data_df["Target"] = data_df["Target"].astype(str)
else:
    st.error("The 'Target' column is missing from the dataset.")
    st.stop()

# Convert all object-type columns to string and numeric columns to float
for col in data_df.columns:
    if data_df[col].dtype == 'O':
        data_df[col] = data_df[col].astype(str).fillna('')
    else:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        
feature_classes = {
    'firstorder': ['Entropy', 'Energy', 'Uniformity', 'MeanAbsoluteDeviation', 'Skewness', 'Kurtosis'],
    'glcm': ['Contrast', 'Idm', 'Correlation', 'ClusterProminence', 'ClusterShade'],
    'gldm': ['GrayLevelNonUniformity', 'DependenceNonUniformity', 'DependenceVariance', 'LowGrayLevelEmphasis', 'HighGrayLevelEmphasis'],
    'glszm': ['ZoneEntropy', 'LargeAreaEmphasis', 'SmallAreaEmphasis', 'GrayLevelVariance', 'SizeZoneNonUniformity'],
    'glrlm': ['RunLengthNonUniformity', 'ShortRunEmphasis', 'LongRunEmphasis']
}

# Generate a list of selected columns based on the feature classes
selected_columns = []
for key, features in feature_classes.items():
    selected_columns.extend([f'original_{key}_{feature}' for feature in features])

selected_columns.append('Target')
        
if page_selection == "Textual Analysis":
    st.title("Textual Analysis")
    st.write("This page includes dataset summaries, descriptive statistics, and hypothesis testing for selected features.")
    
    # Filter option only for textual analysis page
    analysis_type = st.sidebar.selectbox("Select Data Type", options=["Both", "COVID", "Normal"])

    # Load and filter data based on selection
    data_df = load_data()
    if "Target" in data_df.columns:
        data_df["Target"] = data_df["Target"].astype(str)
    else:
        st.error("The 'Target' column is missing from the dataset.")
        st.stop()

    if analysis_type == "COVID":
        data_df = data_df[data_df["Target"] == '1']
    elif analysis_type == "Normal":
        data_df = data_df[data_df["Target"] == '0']

    # Convert all object-type columns to string and numeric columns to float
    for col in data_df.columns:
        if data_df[col].dtype == 'O':
            data_df[col] = data_df[col].astype(str).fillna('')
        else:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    # Show the dataset overview based on selected filter
    dataset_overview(data_df, analysis_type)

    # Display the filtered data preview
    st.markdown(f'<div class="output-container"><h3>Filtered Data for {analysis_type} Cases</h3></div>', unsafe_allow_html=True)
    st.write(data_df.head())

    # Handle empty DataFrame case
    if data_df.empty:
        st.error(f"No data available for {analysis_type} filter. Please verify the dataset or the filter selection.")
        st.stop()

    # Handle missing values and EDA
    data_df = handle_missing_values(data_df)
    # KPI Section
    st.markdown('<div class="main-header">Descriptive Statistics for Selected Feature</div>', unsafe_allow_html=True)

    # Numerical column selection
    numerical_columns = data_df.select_dtypes(include=[np.number]).columns
    if not numerical_columns.empty:
        selected_feature = st.selectbox("Select a Feature for Analysis", options=numerical_columns)

        # Calculate statistics for the selected feature
        mean_value = data_df[selected_feature].mean()
        median_value = data_df[selected_feature].median()
        mode_value = data_df[selected_feature].mode().iloc[0] if not data_df[selected_feature].mode().empty else "No Mode"
        std_dev_value = data_df[selected_feature].std()
        iqr_value = data_df[selected_feature].quantile(0.75) - data_df[selected_feature].quantile(0.25)

        # Display KPI Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="kpi-card">Mean<br>{mean_value:.2f}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-card">Median<br>{median_value:.2f}</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="kpi-card">Mode<br>{mode_value}</div>', unsafe_allow_html=True)

        # Variability Analysis Section
        st.markdown('<div class="main-header">Variability Analysis for Selected Feature</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="kpi-card">Standard Deviation<br>{std_dev_value:.2f}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-card">Interquartile Range (IQR)<br>{iqr_value:.2f}</div>', unsafe_allow_html=True)
    else:
        st.error("No numerical columns available for analysis.")

    # Compute correlation matrix
    corr_matrix = correlation_matrix(data_df, selected_columns)

    # Display correlation matrix in textual format
    st.write("### Correlation Matrix ")
    st.write(corr_matrix)

    # Identify top 5 correlated features with the target
    target_column = 'Target'
    top_features, top_correlations = top_correlated_features(corr_matrix, target_column)
    if top_correlations is not None:
        st.write("### Top 5 Features Most Correlated with Target ")
        st.write(top_correlations[1:])  # Exclude the target column itself

    filtered_data = remove_outliers_isolation_forest(data_df)
        
    display_textual_outlier_comparison(data_df, filtered_data, top_features)

    # Normalize data with Min-Max or Z-score
    normalized_df = normalize_data(data_df, selected_columns, normalization_method)

    transformed_df = apply_transformations(data_df, selected_columns)
    
    normality_results = perform_normality_tests(data_df, transformed_df, selected_columns)
    
    perform_t_test(data_df,top_features)
    
    pca_analysis(data_df)
    
    model_result = train_and_compare_classification_models(data_df, top_features)
    
    show_textual_report(model_result)
    
    

elif page_selection == "Visual Comparisons":
    st.title("Visual Comparisons")
    st.write("This page includes before-and-after visualizations for outlier removal and normalization.")
    col1, col2 = st.columns(2)
    
    with col1:
        corr_matrix = correlation_matrix(data_df, selected_columns)
        # Display heatmap for correlation matrix using plotly
        st.write("### Correlation Matrix Heatmap")
        fig_heatmap = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig_heatmap.update_layout(title="Correlation Matrix for Selected Features")
        st.plotly_chart(fig_heatmap)
        
    with col2:
        top_features, top_correlations = top_correlated_features(corr_matrix)
        # Display bar chart for top 5 correlated features
        if top_correlations is not None:
            st.write("### Top 5 Features Most Correlated with Target ")
            top_features_values = top_correlations[1:]  # Exclude the target column itself
            fig_bar = px.bar(top_features_values, x=top_features_values.index, y=top_features_values.values,
                            labels={'y': 'Correlation', 'index': 'Features'})
            fig_bar.update_layout(title="Top 5 Correlated Features with Target")
            st.plotly_chart(fig_bar)
        
    kpi_and_visualization(data_df, selected_columns)
    
    filtered_data = remove_outliers_isolation_forest(data_df)

    transformed_data = log_transform(filtered_data)
    
    visualize_top_features_box_plots(transformed_data, top_features)
    
    # Normalize data with Min-Max or Z-score
    normalized_df = normalize_data(data_df, selected_columns, normalization_method)
    
    # Plot QQ plots before and after normalization
    plot_qq_before_after_with_plotly(data_df, normalized_df, top_features)
    
    # Plot Histograms Before and After Normalization
    plot_histograms_before_after(data_df, normalized_df, top_features)
    
    pca, pca_df = perform_pca(data_df,n_components=2)
    
    plot_pca_2d(pca_df, data_df)
    
    model_result = train_and_compare_classification_models(data_df, top_features)
    
    show_visualizations(model_result, top_features)
    
