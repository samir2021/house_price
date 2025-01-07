import warnings
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Synthetic generation of data examples for training the model
def generate_house_data(n_samples=100):
    np.random.seed(42)
    size = np.random.normal(1500, 500, n_samples)
    price = size * 100 + np.random.normal(0, 10000, n_samples)
    return pd.DataFrame({'size_sqft': size, 'price': price})

# Function for instantiating and training linear regression model
def train_model():
    df = generate_house_data()
    
    # Train-test data splitting
    X = df[['size_sqft']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

# Streamlit User Interface for Deployed Model
def main():
    # Page configuration
    st.set_page_config(page_title="Smart House Pricer", page_icon="🏠", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {padding: 2rem}
        .stButton>button {width: 100%; background-color: #FF4B4B; color: white;}
        .prediction-box {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.svgrepo.com/show/529278/home-1.svg", width=100)
        st.title("Model Settings")
        confidence = st.slider("Prediction Confidence", 0, 100, 95)
    
    # Main content in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title('🏠 Smart House Pricer')
        st.subheader('Advanced Price Prediction Tool')
        
        # Input section with metrics
        st.markdown("### House Details")
        size = st.number_input('House Size (sq ft)', 
                              min_value=500, 
                              max_value=5000, 
                              value=1500,
                              help="Enter the total square footage of the house")
        
        # Additional features for future expansion
        col3, col4 = st.columns(2)
        with col3:
            bedrooms = st.selectbox('Bedrooms', [1, 2, 3, 4, 5, 6])
        with col4:
            bathrooms = st.selectbox('Bathrooms', [1, 1.5, 2, 2.5, 3, 3.5, 4])
    
    with col2:
        st.markdown("### Market Insights")
        st.metric(label="Average Price/sq ft", value="$100")
        st.metric(label="Market Trend", value="↗️ 5%")
    
    # Prediction section
    if st.button('Calculate Price Estimate'):
        model = train_model()
        prediction = model.predict([[size]])
        
        # Display prediction in a styled container
        st.markdown("### Price Estimation Results")
        with st.container():
            st.markdown(f"""
                <div class='prediction-box'>
                    <h2 style='text-align: center; color: #FF4B4B;'>${prediction[0]:,.2f}</h2>
                    <p style='text-align: center;'>Estimated Price</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Enhanced visualization
        df = generate_house_data()
        fig = px.scatter(df, x='size_sqft', y='price',
                        title='Market Analysis: Size vs Price',
                        labels={'size_sqft': 'House Size (sq ft)', 
                               'price': 'Price ($)'},
                        template='plotly_white')
        fig.add_scatter(x=[size], y=[prediction[0]],
                       mode='markers',
                       marker=dict(size=20, color='red', symbol='star'),
                       name='Your House')
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.markdown("### Additional Insights")
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("Price per sq ft", f"${prediction[0]/size:,.2f}")
        with col6:
            st.metric("Market Position", "Above Average")
        with col7:
            st.metric("Confidence Score", f"{confidence}%")

if __name__ == '__main__':
    main()