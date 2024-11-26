import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def dataset_overview(data,analysis_type):
    
    num_rows = data.shape[0]
    num_columns = data.shape[1]
    
    # Missing values
    missing_values = data.isnull().sum().sum()
    # Additional details at the top
    st.write(f"### Dataset Overview for {analysis_type} Data")
    st.write(f"- **Total Rows:** {num_rows}")
    st.write(f"- **Total Columns:** {num_columns}")
    st.write(f"- **Missing Values:** {missing_values}")
    
    # Numerical columns and rows
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    num_numerical_columns = len(numerical_columns)
    numerical_rows = data[numerical_columns].count().sum()  # Total non-null rows for numerical columns
    
    # Categorical columns and rows
    categorical_columns = data.select_dtypes(include=[object]).columns
    num_categorical_columns = len(categorical_columns)
    categorical_rows = data[categorical_columns].count().sum()  # Total non-null rows for categorical columns
    
    # Summary for numerical columns
    numerical_summary = data[numerical_columns].describe().T[['mean', '50%', 'std']] if not numerical_columns.empty else "No numerical data available"
    
    # Summary for categorical columns
    categorical_summary = data[categorical_columns].describe().T[['top', 'freq']] if not categorical_columns.empty else "No categorical data available"
    
    # Layout for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"### Numerical Features Summary")
        if numerical_columns.empty:
            st.write("No numerical data available.")
        else:
            st.write(numerical_summary)
        st.write(f"**Number of Numerical Columns:** {num_numerical_columns}")
        st.write(f"**Total Non-Null Numerical Rows:** {numerical_rows}")

    with col2:
        st.write(f"### Categorical Features Summary")
        if categorical_columns.empty:
            st.write("No categorical data available.")
        else:
            st.write(categorical_summary)
        st.write(f"**Number of Categorical Columns:** {num_categorical_columns}")
        st.write(f"**Total Non-Null Categorical Rows:** {categorical_rows}")

feature_classes = {
    'firstorder': ['Entropy', 'Energy', 'Uniformity', 'MeanAbsoluteDeviation', 'Skewness', 'Kurtosis'],
    'glcm': ['Contrast', 'Idm', 'Correlation', 'ClusterProminence', 'ClusterShade'],
    'gldm': ['GrayLevelNonUniformity', 'DependenceNonUniformity', 'DependenceVariance', 'LowGrayLevelEmphasis', 'HighGrayLevelEmphasis'],
    'glszm': ['ZoneEntropy', 'LargeAreaEmphasis', 'SmallAreaEmphasis', 'GrayLevelVariance', 'SizeZoneNonUniformity'],
    'glrlm': ['RunLengthNonUniformity', 'ShortRunEmphasis', 'LongRunEmphasis']
}

selected_columns = []
for key, features in feature_classes.items():
    selected_columns.extend([f'original_{key}_{feature}' for feature in features])

selected_columns.append('Target')

def handle_missing_values(data_df):
    missing_values = data_df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        st.success("No missing values detected.")
    else:
        # Layout for side-by-side display: missing values table on the left, handling options on the right
        col1, col2 = st.columns([1, 1])

        with col1:
            # Missing values table
            st.write("### Missing Values in Dataset")
            st.dataframe(missing_values)

        with col2:
            # Missing values handling options
            st.write("### Handle Missing Values")
            handle_missing = st.radio("Choose method to handle missing values:",
                                      ["Fill with 0", "Fill with Mean", "Fill with Median", "Drop Missing Rows", "Do Nothing"])

            if handle_missing == "Fill with 0":
                data_df.fillna(0, inplace=True)
                st.success("Filled missing values with 0.")

            elif handle_missing == "Fill with Mean":
                for col in data_df.columns:
                    if data_df[col].isnull().sum() > 0 and data_df[col].dtype in ['float64', 'int64']:
                        mean_value = data_df[col].mean()
                        data_df[col].fillna(mean_value, inplace=True)
                st.success("Filled missing values with the mean of respective columns.")

            elif handle_missing == "Fill with Median":
                for col in data_df.columns:
                    if data_df[col].isnull().sum() > 0 and data_df[col].dtype in ['float64', 'int64']:
                        median_value = data_df[col].median()
                        data_df[col].fillna(median_value, inplace=True)
                st.success("Filled missing values with the median of respective columns.")

            elif handle_missing == "Drop Missing Rows":
                data_df.dropna(inplace=True)
                st.success("Dropped rows with missing values.")

            else:
                st.info("No changes made.")

    return data_df

def correlation_matrix(data_df, selected_columns):
    # Filter the dataset to include only the selected columns based on feature_classes
    data_df = data_df[selected_columns]
    data_df['Target'] = pd.to_numeric(data_df['Target'], errors='coerce')

    # Only use numeric columns
    numeric_data_df = data_df.select_dtypes(include=[np.number])

    # Handle any columns that may contain tuples or lists as strings
    for col in numeric_data_df.columns:
        if numeric_data_df[col].dtype == 'object':  # Check if column is stored as object
            try:
                numeric_data_df[col] = numeric_data_df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
            except Exception as e:
                st.warning(f"Couldn't convert {col} to numeric values: {e}")

    # Compute the correlation matrix
    corr_matrix = numeric_data_df.corr()

    return corr_matrix

def top_correlated_features(corr_matrix, target_column = 'Target', n=5):
    if target_column in corr_matrix.columns:
        # Get correlation with target and sort
        target_corr = corr_matrix[target_column].sort_values(ascending=False)
        return target_corr.head(n+1).index.tolist(), target_corr.head(n+1)
    else:
        st.error(f"Target column '{target_column}' not found in the correlation matrix.")
        return [], None

def kpi_and_visualization(data_df, selected_columns, target_column = 'Target'):
    # Compute correlation matrix
    corr_matrix = data_df[selected_columns].corr()
    
    # Get top 5 features most correlated with target
    top_features, _ = top_correlated_features(corr_matrix, target_column, n=5)

    st.write(f"### Top 5 Features Correlated with {target_column}")

    # First Row: Histograms for each top feature
    st.write("#### Histograms for Top Correlated Features")
    hist_cols = st.columns(len(top_features))
    for i, feature in enumerate(top_features):
        if feature != target_column:
            hist_fig = px.histogram(data_df, x=feature, nbins=30)
            hist_fig.update_layout(showlegend=False)
            hist_cols[i].plotly_chart(hist_fig, use_container_width=True)

    # Second Row: Box Plots for each top feature
    st.write("#### Box Plots for Top Correlated Features")
    box_cols = st.columns(len(top_features))
    for i, feature in enumerate(top_features):
        if feature != target_column:
            box_fig = px.box(data_df, y=feature)
            box_cols[i].plotly_chart(box_fig, use_container_width=True)
            
    # Third Row: Scatter Plots for each top feature against the target
    st.write(f"#### Scatter Plots of Top Correlated Features vs {target_column}")
    scatter_cols = st.columns(len(top_features))
    for i, feature in enumerate(top_features):
        if feature != target_column:
            scatter_fig = px.scatter(data_df, x=feature, y=target_column)
            scatter_cols[i].plotly_chart(scatter_fig, use_container_width=True)


def remove_outliers_isolation_forest(data_df, iqr_factor = 1.5, max_iterations=10):
    numeric_data = data_df.select_dtypes(include=[np.number])
    for iteration in range(max_iterations):
        initial_row_count = len(numeric_data)
        for column in numeric_data.columns:
            q1 = numeric_data[column].quantile(0.25)
            q3 = numeric_data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (iqr_factor * iqr)
            upper_bound = q3 + (iqr_factor * iqr)
            numeric_data = numeric_data[(numeric_data[column] >= lower_bound) & (numeric_data[column] <= upper_bound)]
        
        if len(numeric_data) == initial_row_count:  # Stop if no more rows are removed
            break
    
    # Return the cleaned data
    filtered_data = data_df.loc[numeric_data.index]
    return filtered_data

def log_transform(data_df):
    numeric_data = data_df.select_dtypes(include=[np.number])
    log_transformed_df = data_df.copy()

    # Apply log transformation only to positive numeric columns
    for column in numeric_data.columns:
        if (numeric_data[column] > 0).all():
            log_transformed_df[column] = np.log1p(numeric_data[column])
    
    return log_transformed_df

def display_textual_outlier_comparison(data_df, filtered_data, top_features):
    st.write("### Outlier Removal Summary ")

    # Filter DataFrames to include only numeric columns
    numeric_data_df = data_df.select_dtypes(include=[np.number])
    numeric_filtered_data = filtered_data.select_dtypes(include=[np.number])

    # Initialize an empty list to collect summary data for each feature
    summary_data = []

    for feature in top_features:
        if feature in numeric_data_df.columns:
            # Compute statistics for the original and filtered data
            original_mean = numeric_data_df[feature].mean()
            filtered_mean = numeric_filtered_data[feature].mean()
            original_median = numeric_data_df[feature].median()
            filtered_median = numeric_filtered_data[feature].median()
            original_std = numeric_data_df[feature].std()
            filtered_std = numeric_filtered_data[feature].std()
            num_outliers_removed = len(numeric_data_df[feature]) - len(numeric_filtered_data[feature])

            # Append feature summary as a dictionary
            summary_data.append({
                "Feature": feature,
                "Original Mean": original_mean,
                "Filtered Mean": filtered_mean,
                "Original Median": original_median,
                "Filtered Median": filtered_median,
                "Original Std Dev": original_std,
                "Filtered Std Dev": filtered_std,
                "Outliers Removed": num_outliers_removed
            })

    # Convert summary data to a DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Display the summary table in Streamlit
    st.write(summary_df)

def visualize_top_features_box_plots(filtered_data, top_features):

    st.write("#### Box Plots for Top Features After Outlier Removal")

    # Only include numeric columns in filtered data
    numeric_filtered_data = filtered_data.select_dtypes(include=[np.number])

    # Create columns for each top feature box plot
    box_cols = st.columns(len(top_features))
    
    for i, feature in enumerate(top_features):
        if feature in numeric_filtered_data.columns:
            # Create a box plot for each feature after outlier removal
            box_fig = px.box(numeric_filtered_data, y=feature)
            box_cols[i].plotly_chart(box_fig, use_container_width=True)

def normalize_data(data_df, selected_columns, normalization_method='Min-Max', is_textual=False):
    """
    Normalize the selected columns in the dataset using the specified method. Optionally handle textual data.

    Parameters:
    - data_df: The DataFrame to normalize.
    - selected_columns: List of columns to normalize.
    - normalization_method: The normalization method ('Min-Max', 'Z-score', 'Max-Abs').
    - is_textual: Flag to indicate if the columns contain textual data.

    Returns:
    - normalized_df: A DataFrame with normalized columns.
    """
    normalized_df = data_df.copy()

    # Select numeric columns only if not handling textual data
    if not is_textual:
        numeric_columns = [col for col in selected_columns if pd.api.types.is_numeric_dtype(data_df[col])]
    else:
        numeric_columns = selected_columns

    if normalization_method == 'Min-Max':
        for col in numeric_columns:
            min_val = data_df[col].min()
            max_val = data_df[col].max()
            normalized_df[col] = (data_df[col] - min_val) / (max_val - min_val)
        st.success("Applied Min-Max Normalization.")

    elif normalization_method == 'Z-score':
        for col in numeric_columns:
            mean = data_df[col].mean()
            std = data_df[col].std()
            normalized_df[col] = (data_df[col] - mean) / std
        st.success("Applied Z-score Normalization.")

    elif normalization_method == 'Max-Abs':
        for col in numeric_columns:
            max_abs = data_df[col].abs().max()
            normalized_df[col] = data_df[col] / max_abs
        st.success("Applied Max-Abs Normalization.")

    elif is_textual:
        # Convert textual data to numeric form using TF-IDF (or any other method)
        vectorizer = TfidfVectorizer(max_features=100)
        transformed_text_data = vectorizer.fit_transform(data_df[selected_columns[0]]).toarray()
        transformed_df = pd.DataFrame(transformed_text_data, columns=vectorizer.get_feature_names_out())
        normalized_df[selected_columns[0]] = transformed_df.mean(axis=1)  # Example: Combine features
        st.success("Applied Textual Data Normalization (TF-IDF).")

    else:
        st.error(f"Unknown normalization method: {normalization_method}")

    return normalized_df

def preprocess_data(data_df, selected_columns):
    """
    Preprocess data by converting columns to numeric and handling non-numeric values.
    """
    # Filter out non-numeric columns
    numeric_columns = []
    for col in selected_columns:
        try:
            # Attempt to convert the column to numeric
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
            numeric_columns.append(col)
        except Exception as e:
            print(f"Skipping non-numeric column: {col}, Error: {e}")

    # Drop columns with all NaN values
    data_df = data_df.dropna(axis=1, how='all')

    # Fill remaining NaN values with the median of numeric columns
    data_df.fillna(data_df[numeric_columns].median(), inplace=True)

    return data_df[numeric_columns]

def apply_transformations(data_df, selected_columns):
    """
    Apply log and Yeo-Johnson transformations to the selected columns.
    """
    # Preprocess the data first
    numeric_df = preprocess_data(data_df, selected_columns)

    transformed_data = numeric_df.copy()

    # Apply log transformation (for positive-valued features)
    for col in numeric_df.columns:
        if (numeric_df[col] > 0).all():  # Check if all values are positive
            transformed_data[col] = np.log1p(numeric_df[col])

    # Apply Yeo-Johnson transformation
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    transformed_data = pd.DataFrame(
        pt.fit_transform(transformed_data),
        columns=transformed_data.columns
    )

    return transformed_data

def perform_normality_tests(data_df, normalized_df, selected_columns):
    """
    Perform Anderson-Darling and KS tests to check for normality before and after normalization.
    """
    # Prepare a list to store results
    results = []

    # Loop over the selected columns
    for col in selected_columns:
        if pd.api.types.is_numeric_dtype(data_df[col]):
            # Initialize a dictionary to store results for each feature
            test_results = {'Feature': col}

            # Anderson-Darling Test (Before Normalization)
            stat = stats.anderson(data_df[col].dropna(), dist='norm')[0]
            critical_value = stats.anderson(data_df[col].dropna(), dist='norm')[1][2]  # Use 5% significance level
            test_results['Anderson-Darling (Before) Stat'] = stat
            test_results['Anderson-Darling (Before) Result'] = 'Normal' if stat < critical_value else 'Not Normal'

            # Anderson-Darling Test (After Normalization)
            stat = stats.anderson(normalized_df[col].dropna(), dist='norm')[0]
            critical_value = stats.anderson(normalized_df[col].dropna(), dist='norm')[1][2]  # Use 5% significance level
            test_results['Anderson-Darling (After) Stat'] = stat
            test_results['Anderson-Darling (After) Result'] = 'Normal' if stat < critical_value else 'Not Normal'

            # KS Test (Before Normalization) against a fitted normal distribution
            mean, std = data_df[col].mean(), data_df[col].std()
            d_stat, p_value = stats.kstest(data_df[col].dropna(), 'norm', args=(mean, std))
            test_results['KS Test (Before) p-value'] = p_value
            test_results['KS Test (Before) Result'] = 'Normal' if p_value > 0.05 else 'Not Normal'

            # KS Test (After Normalization) against a fitted normal distribution
            mean, std = normalized_df[col].mean(), normalized_df[col].std()
            d_stat, p_value = stats.kstest(normalized_df[col].dropna(), 'norm', args=(mean, std))
            test_results['KS Test (After) p-value'] = p_value
            test_results['KS Test (After) Result'] = 'Normal' if p_value > 0.05 else 'Not Normal'

            # Append the results for the current feature
            results.append(test_results)

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Display the results as a table
    st.write("### Normality Test Results Comparison")
    st.write(results_df)
    
def plot_histograms_before_after(data_df, normalized_df, top_features):
    """
    Create histograms before and after normalization side by side for each feature.
    """
    st.write(f"#### Histogram for Top Features Before And After Normalization")
    # Create two rows with equal number of columns
    row1 = st.columns(len(top_features))  # Columns for histograms before normalization
    row2 = st.columns(len(top_features))  # Columns for histograms after normalization

    for i, col in enumerate(top_features):
        if pd.api.types.is_numeric_dtype(data_df[col]):
            # Create histogram for original data
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=data_df[col], name=f'{col} (Original)', opacity=0.6, nbinsx=50))

            # Create histogram for normalized data
            fig.add_trace(go.Histogram(x=normalized_df[col], name=f'{col} (Normalized)', opacity=0.6, nbinsx=50))

            # Update layout for histogram
            fig.update_layout(
                barmode='overlay',
                xaxis_title=col,
                yaxis_title='Frequency',
                template='plotly_dark'
            )

            # Display histograms before and after in side-by-side columns
            row1[i].plotly_chart(fig)

def plot_qq_before_after_with_plotly(data_df, normalized_df, top_features):
    """
    Create QQ Plots before and after normalization side by side using Plotly.
    """
    
    st.write(f"#### QQ Plot for Top Features Before And After Normalization")
    # Create two rows with equal number of columns
    row1 = st.columns(len(top_features))  # Columns for QQ plots before normalization
    row2 = st.columns(len(top_features))  # Columns for QQ plots after normalization

    for i, col in enumerate(top_features):
        if pd.api.types.is_numeric_dtype(data_df[col]):
            # Create QQ plot for original data
            fig = px.scatter(x=np.sort(data_df[col]), y=np.sort(stats.norm.rvs(size=len(data_df[col]))),  
                             labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})

            # Add a line for normal distribution
            fig.add_scatter(x=np.sort(data_df[col]), y=np.sort(data_df[col]), mode='lines', name='Normal Distribution')

            # Display QQ plot before normalization
            row1[i].plotly_chart(fig)

            # Create QQ plot for normalized data
            fig = px.scatter(x=np.sort(normalized_df[col]), y=np.sort(stats.norm.rvs(size=len(normalized_df[col]))), 
                             labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})

            # Add a line for normal distribution
            fig.add_scatter(x=np.sort(normalized_df[col]), y=np.sort(normalized_df[col]), mode='lines', name='Normal Distribution')

            # Display QQ plot after normalization
            row2[i].plotly_chart(fig)

def perform_t_test(data_df, top_features, target_column='Target'):
    st.write("### Hypothesis Testing Using Two-Sample t-Test")
    st.write("""
            ### Null Hypothesis (H₀):
            - **H₀**: There is no significant difference in the means of the feature between the two groups (Normal cases and COVID cases).
            - Mathematically, this is expressed as:  
            𝜇₁ = 𝜇₂  
            where 𝜇₁ and 𝜇₂ are the means of the feature for the two groups.

            ### Alternative Hypothesis (H₁):
            - **H₁**: There is a significant difference in the means of the feature between the two groups (Normal cases and COVID cases).
            - Mathematically, this is expressed as:  
            𝜇₁ ≠ 𝜇₂  
            where 𝜇₁ and 𝜇₂ are the means of the feature for the two groups.
            """)

    # Check if 'Target' column exists in the DataFrame
    if target_column not in data_df.columns:
        st.error(f"Error: The column '{target_column}' does not exist in the DataFrame.")
        return

    # Separate the data into two groups based on the target column
    group1 = data_df[data_df[target_column] == 0]  # Normal cases
    group2 = data_df[data_df[target_column] == 1]  # COVID cases

    # Store results
    t_test_results = []

    # Exclude the target column from the list of features for t-test
    features_for_test = [feature for feature in top_features if feature != target_column]

    for feature in features_for_test:
        try:
            # Drop NaN values for the feature
            group1_values = group1[feature].dropna()
            group2_values = group2[feature].dropna()

            # Check for empty groups
            if group1_values.empty or group2_values.empty:
                st.warning(f"No data available for feature '{feature}' in one of the groups.")
                t_stat, p_value = np.nan, np.nan
            else:
                # Check for zero variance in the groups
                if group1_values.var() == 0 or group2_values.var() == 0:
                    st.warning(f"Zero variance detected for feature '{feature}' in one of the groups.")
                    t_stat, p_value = np.nan, np.nan
                else:
                    # Perform two-sample t-test
                    t_stat, p_value = stats.ttest_ind(group1_values, group2_values, equal_var=False)

            # Determine significance
            significance = "Reject H0" if p_value < 0.05 else "Fail to Reject H0"

            # Append results
            t_test_results.append({
                'Feature': feature,
                't-Statistic': t_stat,
                'p-Value': p_value,
                'Result': significance
            })

        except Exception as e:
            st.error(f"Error performing t-test on feature '{feature}': {e}")
            t_stat, p_value = np.nan, np.nan
            t_test_results.append({
                'Feature': feature,
                't-Statistic': t_stat,
                'p-Value': p_value,
                'Result': "Error"
            })

    # Create a DataFrame for displaying results
    t_test_df = pd.DataFrame(t_test_results)

    # Display the results as a comparison table
    st.write("### Comparison of t-Test Results")
    st.write(t_test_df)

    return t_test_df
            
def perform_pca(data_df, n_components=2):
    """
    Perform PCA on the dataset and return the transformed data for visualization.
    """
    # Drop non-numeric columns and handle missing values
    data_numeric = data_df.select_dtypes(include=[np.number]).dropna()
    
    # Standardize the data (mean = 0, variance = 1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_numeric)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create a DataFrame with the PCA components
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Explained variance ratio
    return pca, pca_df

def plot_pca_2d(pca_df, data_df, target_column='Target'):
    """
    Visualize the PCA results in a 2D scatter plot.
    """
    
    st.write("### PCA Analysis Graph")
    # Add the target column for coloring the points
    pca_df[target_column] = data_df[target_column]
    
    # Plotting the first two principal components
    fig = px.scatter(pca_df, x='PC1', y='PC2', color=target_column,
                     title="PCA - 2D Projection",
                     labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
    st.plotly_chart(fig)
    
def pca_analysis(data_df, target_column='Target', n_components=2):
    st.write("### PCA Analysis")
    
    # Perform PCA with the specified number of components
    pca, pca_df = perform_pca(data_df, n_components)
    
    # Create a DataFrame to display the explained variance ratio for each component and the total variance
    explained_variance = {
        'Principal Component': [f'PC{i+1}' for i in range(n_components)] + ['Total Variance Explained'],
        'Variance Explained': list(pca.explained_variance_ratio_) + [sum(pca.explained_variance_ratio_)]
    }
    
    # Create a DataFrame to represent the explained variance ratios
    explained_variance_df = pd.DataFrame(explained_variance)
    
    # Display the explained variance ratio as a table
    st.write("### Explained Variance Ratio Table:")
    st.write(explained_variance_df)
    
def train_and_compare_classification_models(data_df, top_features, target_column='Target'):
    # Handle missing values and select the top features
    data_clean = data_df.dropna(subset=top_features + [target_column])

    # Separate features (X) and target (y)
    X = data_clean[top_features]
    y = data_clean[target_column]

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_test = y_test.astype(int)
    models = {
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    

    model_results = {}

    for model_name, model in models.items():
        # Cross-validation to evaluate the model more robustly
        cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        # Train the model on the entire training data
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)
        y_pred = y_pred.astype(int) 
        y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Store results for later use in visualization page
        model_results[model_name] = {
            'model': model,  # Store the model itself
            'cross_val_scores': cross_val_scores,
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'conf_matrix': conf_matrix,
            'y_prob': y_prob,
            'y_pred': y_pred,
            'y_test': y_test
        }

    return model_results

def show_textual_report(model_results):
    model_comparison_data = []

    # Iterate through the model results and collect the data for each model
    for model_name, results in model_results.items():
        model_comparison_data.append({
            "Model": model_name,
            "Cross-Validation Scores": ', '.join([f"{score:.4f}" for score in results['cross_val_scores']]),
            "Mean Cross-Validation Accuracy": f"{results['cross_val_scores'].mean():.4f}",
            "Accuracy": f"{results['accuracy']:.4f}",
            "Classification Report": str(results['classification_report'])
        })

    # Convert the collected data into a DataFrame for better display
    model_comparison_df = pd.DataFrame(model_comparison_data)

    # Display the comparison table
    st.write("### Classification Model Comparison")
    st.dataframe(model_comparison_df)  # Using dataframe to make it interactive

def show_visualizations(model_results, top_features):
    st.write("### Classification Model Visualizations")

    # Create 3 columns layout
    col1, col2, col3 = st.columns(3)

    for model_name, results in model_results.items():
        # Ensure y_true is in binary numeric format (0, 1)
        y_test = results['y_test'].astype(int)
        y_prob = results['y_prob']
        
        with col1:  # First column for ROC Curve
            st.write(f"####  {model_name}")
            try:
                fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)  # Specify pos_label explicitly as 1
                roc_auc = auc(fpr, tpr)

                # Plot ROC curve
                fig, ax = plt.subplots(figsize=(6, 5.5))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'Receiver Operating Characteristic (ROC) Curve: {model_name}')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            except ValueError as e:
                st.error(f"Error calculating ROC Curve: {e}")
        
        with col2:  # Second column for Classification Report and Confusion Matrix
            # Confusion Matrix
            st.write("#### Confusion Matrix:")
            conf_matrix = results['conf_matrix']

            # Plot confusion matrix using seaborn
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "COVID"], yticklabels=["Normal", "COVID"])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f"Confusion Matrix: {model_name}")
            st.pyplot(fig)
        
        with col3:  # Third column for feature importance (only for Random Forest)
            if model_name != 'Logistic Regression':  # Feature importance is not available for Logistic Regression
                # Feature importance (only for Random Forest)
                if model_name == 'Random Forest':
                    feature_importance = results['model'].feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'Feature': top_features,
                        'Importance': feature_importance
                    }).sort_values(by='Importance', ascending=False)

                    st.write(f"#### Feature Importance for {model_name}:")
                    st.write(feature_importance_df)

                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
                    ax.set_title(f"Feature Importance: {model_name}")
                    st.pyplot(fig)
    with col1:
        st.empty()
    with col2:
        st.empty()
    with col3:
        st.empty()
    

        