import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# --- 1. Page Configuration ---
st.set_page_config(page_title="2026 Winter Olympics Predictor", page_icon="❄️")

# --- 2. Caching the Data & Model ---
# We use cache so we don't reload/retrain every time you click a button
@st.cache_resource
def load_and_train_model():
    # A. Load Data
    # Make sure these CSV files are in the same folder as app.py
    try:
        medals_df = pd.read_csv('olympic_medals.csv')
        hosts_df = pd.read_csv('olympic_hosts.csv')
    except FileNotFoundError:
        st.error("❌ CSV files not found! Please place 'olympic_medals.csv' and 'olympic_hosts.csv' in the same folder.")
        st.stop()

    # B. Preprocessing (Recreating your notebook logic)
    # 1. Filter for Winter Games
    winter_hosts = hosts_df[hosts_df['game_season'] == 'Winter'].copy()
    winter_slugs = winter_hosts['game_slug'].unique()
    medals_df = medals_df[medals_df['slug_game'].isin(winter_slugs)].copy()
    
    # 2. Merge Year
    # Drop duplicates in hosts to avoid merge explosion
    winter_hosts = winter_hosts.drop_duplicates(subset='game_slug')
    medals_df = medals_df.merge(winter_hosts[['game_slug', 'game_year']], left_on='slug_game', right_on='game_slug', how='left')
    medals_df.rename(columns={'game_year': 'Year'}, inplace=True)
    
    # 3. Filter >= 1992
    medals_df = medals_df[medals_df['Year'] >= 1992]

    # 4. Aggregate Medals
    country_medals = medals_df.groupby(['country_3_letter_code', 'Year']).size().reset_index(name='Total_Medals')
    country_medals.rename(columns={'country_3_letter_code': 'country_code'}, inplace=True)
    
    # 5. Feature Engineering (Lag Features)
    country_medals = country_medals.sort_values(['country_code', 'Year'])
    country_medals['Medals_Prev_1_Games'] = country_medals.groupby('country_code')['Total_Medals'].shift(1).fillna(0)
    country_medals['Medals_Prev_2_Games'] = country_medals.groupby('country_code')['Total_Medals'].shift(2).fillna(0)

    # 6. Add Host Feature for Training
    # Simplified host map for major Winter games
    host_map = {
        2022: 'CHN', 2018: 'KOR', 2014: 'RUS', 2010: 'CAN', 
        2006: 'ITA', 2002: 'USA', 1998: 'JPN', 1994: 'NOR', 1992: 'FRA'
    }
    country_medals['Is_Host'] = country_medals.apply(lambda x: 1 if host_map.get(x['Year']) == x['country_code'] else 0, axis=1)

    # C. Train Model
    # Train on history (up to 2018 or 2022)
    # Using rows where we have history
    train_data = country_medals.dropna()
    
    features = ['Medals_Prev_1_Games', 'Medals_Prev_2_Games', 'Is_Host']
    X = train_data[features]
    y = train_data['Total_Medals']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # D. Prepare 2026 Input Data
    # Get the latest data (2022) to use as input for 2026
    data_2022 = country_medals[country_medals['Year'] == 2022].copy()
    
    X_2026 = pd.DataFrame()
    X_2026['country_code'] = data_2022['country_code']
    X_2026['Medals_Prev_1_Games'] = data_2022['Total_Medals']
    X_2026['Medals_Prev_2_Games'] = data_2022['Medals_Prev_1_Games']
    # Italy is Host in 2026
    X_2026['Is_Host'] = X_2026['country_code'].apply(lambda x: 1 if x == 'ITA' else 0)
    
    # Add Names for display
    code_map = medals_df[['country_3_letter_code', 'country_name']].drop_duplicates().set_index('country_3_letter_code')
    X_2026['country_name'] = X_2026['country_code'].map(code_map['country_name'])

    return model, X_2026

# --- 3. Main App Layout ---

st.title("❄️ 2026 Winter Olympics Predictor")
st.markdown("""
This app predicts the medal count for the 2026 Winter Olympics in **Milano-Cortina, Italy**.
It trains a **Random Forest** model on historical data (1992-2022) in real-time.
""")

# Load resources
with st.spinner('Training model and processing data...'):
    model, data_2026 = load_and_train_model()

# Sidebar for controls
st.sidebar.header("Configuration")
top_n = st.sidebar.slider("Number of Countries to Show", 5, 25, 10)
run_btn = st.sidebar.button("Run Prediction", type="primary")

if run_btn:
    # --- 4. Prediction Logic (THE FIX) ---
    
    # CRITICAL: Select ONLY the columns the model was trained on
    feature_cols = ['Medals_Prev_1_Games', 'Medals_Prev_2_Games', 'Is_Host']
    X_input = data_2026[feature_cols]
    
    # Predict
    predictions = model.predict(X_input)
    
    # Save results to a clean dataframe
    results = data_2026.copy()
    results['Predicted_Medals'] = np.round(predictions).astype(int)
    
    # Sort and Filter
    results = results.sort_values(by='Predicted_Medals', ascending=False).reset_index(drop=True)
    top_results = results.head(top_n)

    # --- 5. Display Results ---
    
    # Metrics
    st.subheader("🏆 Top Contenders")
    col1, col2, col3 = st.columns(3)
    if len(top_results) >= 3:
        col1.metric("🥇 1st Place", f"{top_results.iloc[0]['country_name']}", f"{top_results.iloc[0]['Predicted_Medals']} Medals")
        col2.metric("🥈 2nd Place", f"{top_results.iloc[1]['country_name']}", f"{top_results.iloc[1]['Predicted_Medals']} Medals")
        col3.metric("🥉 3rd Place", f"{top_results.iloc[2]['country_name']}", f"{top_results.iloc[2]['Predicted_Medals']} Medals")

    # Chart
    st.subheader(f"Top {top_n} Predicted Medal Counts")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_results, x='Predicted_Medals', y='country_name', palette='viridis', ax=ax)
    ax.set_xlabel("Predicted Total Medals")
    ax.set_ylabel("")
    st.pyplot(fig)

    # Data Table
    st.subheader("Detailed Data")
    st.dataframe(top_results[['country_name', 'Predicted_Medals', 'Medals_Prev_1_Games', 'Is_Host']])