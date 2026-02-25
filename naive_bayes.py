import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
 
# Page configuration
st.set_page_config(
    page_title="Naive Bayes Classifier",
    page_icon="🤖",
    layout="wide"
)
 
# Title and description
st.title("🤖 Naive Bayes Classification App")
st.markdown("---")
 
# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
 
# Sidebar - Dataset Input
with st.sidebar:
    st.header("📁 Dataset Input")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with your dataset"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! Shape: {st.session_state.df.shape}")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.session_state.df = None
 
# Main content
if st.session_state.df is not None:
    # Dataset preview
    st.header("📊 Dataset Preview")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", st.session_state.df.shape[0])
    with col2:
        st.metric("Columns", st.session_state.df.shape[1])
    with col3:
        missing_values = st.session_state.df.isnull().sum().sum()
        if missing_values > 0:
            st.metric("Missing Values", missing_values, delta="⚠️ Needs attention")
        else:
            st.metric("Missing Values", missing_values)
    
    st.markdown("---")
    
    # Data Processing Section
    st.header("⚙️ Data Processing")
    
    # Target column selection
    target_column = st.selectbox(
        "Select Target Column (y)",
        options=st.session_state.df.columns,
        help="Choose the column you want to predict"
    )
    
    # Check if target column is suitable for classification
    unique_values = st.session_state.df[target_column].nunique()
    total_samples = len(st.session_state.df)
    
    # Validate target column
    is_valid_target = True
    validation_message = ""
    
    if unique_values < 2:
        is_valid_target = False
        validation_message = "❌ Target column must have at least 2 classes for classification"
    elif unique_values > total_samples * 0.5:  # If more than 50% unique values, probably not classification
        is_valid_target = False
        validation_message = "❌ Too many unique values. This looks like regression, not classification"
    elif unique_values > 20:
        st.warning(f"⚠️ Target has {unique_values} classes. Multi-class classification detected")
    
    if not is_valid_target:
        st.error(validation_message)
        st.info("💡 Please select a different column with categorical/label data for classification")
        st.stop()
    
    # Feature columns (all except target)
    feature_columns = [col for col in st.session_state.df.columns if col != target_column]
    
    if len(feature_columns) == 0:
        st.error("❌ No feature columns available! Please select a different target column.")
        st.stop()
    
    st.success(f"✅ Naive Bayes Classification detected! Target: '{target_column}' with {unique_values} classes")
    st.info(f"📊 Feature columns (X): {', '.join(feature_columns[:5])}{'...' if len(feature_columns) > 5 else ''}")
    
    # Handle missing values
    if st.session_state.df.isnull().sum().sum() > 0:
        st.warning("⚠️ Dataset contains missing values")
        missing_option = st.radio(
            "Handle missing values:",
            ["Drop rows with missing values", "Fill with mean/mode"],
            help="Choose how to handle missing values"
        )
        
        if st.button("Apply Missing Value Handling"):
            try:
                if missing_option == "Drop rows with missing values":
                    st.session_state.df = st.session_state.df.dropna()
                    st.success("✅ Dropped rows with missing values")
                else:
                    for col in st.session_state.df.columns:
                        if st.session_state.df[col].dtype in ['int64', 'float64']:
                            st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].mean())
                        else:
                            st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].mode()[0])
                    st.success("✅ Filled missing values with mean/mode")
            except Exception as e:
                st.error(f"❌ Error handling missing values: {str(e)}")
    
    st.markdown("---")
    
    # Model Selection and Training Section
    st.header("🎯 Model Training")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Model selection
        model_type = st.selectbox(
            "Select Naive Bayes Model",
            ["Gaussian Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes"],
            help="Choose the type of Naive Bayes classifier"
        )
        
        # Model description
        if model_type == "Gaussian Naive Bayes":
            st.caption("📝 Best for continuous/numerical features")
        elif model_type == "Multinomial Naive Bayes":
            st.caption("📝 Best for discrete features (counts, frequencies)")
        else:
            st.caption("📝 Best for binary features")
    
    with col2:
        # Train-test split
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data to use for testing"
        ) / 100
    
    with col3:
        # Train button
        train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
    
    if train_button:
        try:
            # Prepare data
            X = st.session_state.df[feature_columns].copy()
            y = st.session_state.df[target_column].copy()
            
            # Check class distribution
            class_counts = y.value_counts()
            min_class_size = class_counts.min()
            
            if min_class_size < 2:
                st.error(f"❌ Cannot perform classification: One or more classes have only {min_class_size} sample(s). Each class needs at least 2 samples for train-test split.")
                st.stop()
            
            # Encode categorical features in X
            le_dict = {}
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    le_dict[col] = le
            
            # Encode target if categorical
            if y.dtype == 'object':
                target_le = LabelEncoder()
                y = target_le.fit_transform(y.astype(str))
                st.info(f"✅ Target encoded: {dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))}")
            
            # Split data with stratification
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except ValueError as e:
                if "The least populated class in y has only 1 member" in str(e):
                    st.error("❌ Cannot perform stratified split: Some classes have only 1 sample. Try a different target column or use a larger dataset.")
                else:
                    st.error(f"❌ Error during data split: {str(e)}")
                st.stop()
            
            # Initialize and configure model based on type
            if model_type == "Gaussian Naive Bayes":
                model = GaussianNB()
                
            elif model_type == "Multinomial Naive Bayes":
                # Check for negative values
                if (X_train < 0).any().any() or (X_test < 0).any().any():
                    st.warning("⚠️ Negative values detected. Taking absolute values for MultinomialNB.")
                    X_train = np.abs(X_train)
                    X_test = np.abs(X_test)
                model = MultinomialNB()
                
            else:  # Bernoulli Naive Bayes
                # Check if features are binary, if not convert
                if (X_train.nunique() > 2).any():
                    st.warning("⚠️ Non-binary features detected. Binarizing using median.")
                    threshold = X_train.median()
                    X_train = (X_train > threshold).astype(int)
                    X_test = (X_test > threshold).astype(int)
                model = BernoulliNB()
            
            # Train model
            with st.spinner("Training model..."):
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate accuracy
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_test_pred)
            
            # Store in session state
            st.session_state.model_trained = True
            st.session_state.model = model
            st.session_state.train_accuracy = train_accuracy
            st.session_state.test_accuracy = test_accuracy
            st.session_state.cm = cm
            st.session_state.model_type = model_type
            st.session_state.target_column = target_column
            st.session_state.feature_columns = feature_columns
            st.session_state.n_classes = len(np.unique(y))
            
            st.success("✅ Model trained successfully!")
            
        except Exception as e:
            error_message = str(e)
            if "The least populated class in y has only 1 member" in error_message:
                st.error("❌ Cannot perform classification: Some classes have only 1 sample. Please select a different target column with balanced classes.")
            else:
                st.error(f"❌ Error during training: {error_message[:100]}...")  # Show only first 100 chars
            st.session_state.model_trained = False
    
    # Display results if model is trained
    if st.session_state.model_trained:
        st.markdown("---")
        st.header("📈 Training Results")
        
        # Accuracy metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Training Accuracy",
                f"{st.session_state.train_accuracy:.2%}",
                help="Accuracy on training data"
            )
        with col2:
            st.metric(
                "Testing Accuracy",
                f"{st.session_state.test_accuracy:.2%}",
                help="Accuracy on testing data"
            )
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get unique classes
        if st.session_state.n_classes <= 10:  # Show labels only if not too many classes
            sns.heatmap(st.session_state.cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        else:
            sns.heatmap(st.session_state.cm, annot=False, fmt='d', cmap='Blues', ax=ax)
            st.caption(f"📝 {st.session_state.n_classes} classes detected, annotations hidden for clarity")
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {st.session_state.model_type}')
        st.pyplot(fig)
        plt.close()
        
        # Model information
        with st.expander("View Model Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Model Information:**")
                st.json({
                    "model_type": st.session_state.model_type,
                    "target_column": st.session_state.target_column,
                    "n_features": len(st.session_state.feature_columns),
                    "n_classes": st.session_state.n_classes
                })
            with col2:
                st.markdown("**Feature Information:**")
                feature_info = {}
                for i, col in enumerate(st.session_state.feature_columns[:10]):
                    feature_info[f"Feature {i+1}"] = col
                if len(st.session_state.feature_columns) > 10:
                    feature_info["..."] = f"and {len(st.session_state.feature_columns)-10} more"
                st.json(feature_info)
 
else:
    # No data uploaded
    st.info("👈 Please upload a CSV file to begin")
    
    # Example format
    with st.expander("📋 Expected CSV Format"):
        st.markdown("""
        Your CSV file should contain:
        - Multiple feature columns (X) - can be numerical or categorical
        - One target column (y) - for classification prediction
        
        **Example (Iris Dataset):**
        ```
        sepal_length,sepal_width,petal_length,petal_width,species
        5.1,3.5,1.4,0.2,setosa
        4.9,3.0,1.4,0.2,setosa
        6.2,3.4,5.4,2.3,virginica
        5.9,3.0,5.1,1.8,virginica
        ```
        
        **Example (Titanic Dataset):**
        ```
        pclass,age,sibsp,parch,fare,sex,embarked,survived
        1,29,0,0,211.5,female,S,1
        2,32,0,0,73.5,male,S,0
        3,26,1,1,14.4,male,S,0
        ```
        
        **Requirements for Classification:**
        - Target column should have 2-20 classes
        - Each class should have at least 2 samples
        - Features can be numerical or categorical
        """)
 
# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and scikit-learn")