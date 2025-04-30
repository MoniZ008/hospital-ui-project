import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Hospital Bed and Resource Forecasting", layout="wide")
st.image("uic.png", use_container_width=True)

st.markdown("""
<div style="border: 2px solid #1a73e8; padding:10px; border-radius:6px; color:#1a73e8; background-color: rgba(230, 240, 255, 0.3); font-weight:600;">
🖥️ Try desktop for best viewing!
</div>
""", unsafe_allow_html=True)

# ------------------ Load Data ------------------
data_path = "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW__20250220.csv"
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['year'] = df['date'].dt.year

# Create grouped version for some visualizations
df_grouped = df.groupby('date').sum(numeric_only=True)

# Categorization Columns
columns_needed = [
    "inpatient_beds_utilization",
    "critical_staffing_shortage_today_yes",
    "critical_staffing_shortage_today_no",
    "percent_of_inpatients_with_covid"
]
if all(col in df.columns for col in columns_needed):
    df["staffing_shortage_ratio"] = df["critical_staffing_shortage_today_yes"] / (
        df["critical_staffing_shortage_today_yes"] + df["critical_staffing_shortage_today_no"]
    )
    df["inpatient_beds_utilization"] *= 100
    df["staffing_shortage_ratio"] *= 100
    df["percent_of_inpatients_with_covid"] *= 100

    def categorize_hospital_pressure(value):
        if value < 60:
            return "Green (Low)"
        elif 60 <= value <= 80:
            return "Yellow (Moderate)"
        else:
            return "Red (High)"

    def categorize_staffing(value):
        if value < 20:
            return "Low"
        elif 20 <= value <= 50:
            return "Moderate"
        else:
            return "High"

    def categorize_covid_burden(value):
        if value < 5:
            return "Low"
        elif 5 <= value <= 15:
            return "Moderate"
        else:
            return "High"

    df["Hospital_Pressure_Category"] = df["inpatient_beds_utilization"].apply(categorize_hospital_pressure)
    df["Staffing_Crisis_Level"] = df["staffing_shortage_ratio"].apply(categorize_staffing)
    df["COVID_Burden_Level"] = df["percent_of_inpatients_with_covid"].apply(categorize_covid_burden)

# ------------------ Train Models On The Fly ------------------
@st.cache_resource
def train_label_encoder(df):
    le = LabelEncoder()
    le.fit(df['state'])
    return le

@st.cache_resource
def train_bed_utilization_model(df, _encoder):
    df = df.copy()
    df['state_encoded'] = _encoder.transform(df['state'])

    required_cols = ['admission_rate', 'emergency_surge', 'staff_current_shortage',
                     'staff_anticipated_shortage', 'available_beds']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0  

    features = [
        'state_encoded', 'week_of_year', 'year', 'admission_rate',
        'emergency_surge', 'staff_current_shortage',
        'staff_anticipated_shortage', 'available_beds'
    ]
    df = df.dropna(subset=['inpatient_beds_utilization'])  
    X = df[features]
    y = df['inpatient_beds_utilization']

    model = XGBRegressor()
    model.fit(X, y)
    return model

@st.cache_resource
def train_staff_shortage_model(df, _encoder):
    df = df.copy()
    df['state_encoded'] = _encoder.transform(df['state'])

    required_cols = ['admission_rate', 'emergency_surge', 'staff_current_shortage',
                     'staff_anticipated_shortage', 'available_beds']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Staffing_shortage_ratio as target
    df['target_staffing'] = df['critical_staffing_shortage_today_yes'] / (
        df['critical_staffing_shortage_today_yes'] + df['critical_staffing_shortage_today_no']
    )

    df = df.dropna(subset=['target_staffing'])  # remove rows with NaN labels
    features = [
        'state_encoded', 'week_of_year', 'year', 'admission_rate',
        'emergency_surge', 'staff_current_shortage',
        'staff_anticipated_shortage', 'available_beds'
    ]
    X = df[features]
    y = df['target_staffing']

    model = XGBRegressor()
    model.fit(X, y)
    return model

# ------------------ Navigation ------------------
section = st.sidebar.radio("Navigation", ["Overview","Inpatient Bed Utilization", "Staffing Shortage Status", "Scenario Planning"])

# ------------------ Overview ------------------
if section == "Overview":
    st.header("📊 Overview")

    # 1. Hospital Bed Utilization Over Time
    st.subheader("Hospital Bed Utilization Over Time")
    fig1, ax1 = plt.subplots(figsize=(12, 5))

    sns.lineplot(data=df, x="date", y="inpatient_beds_utilization", label="Inpatient Bed Utilization", ax=ax1)
    sns.lineplot(data=df, x="date", y="staffing_shortage_ratio", label="Staffing Shortage Ratio", linestyle="dashed", ax=ax1)
    sns.lineplot(data=df, x="date", y="percent_of_inpatients_with_covid", label="% COVID Inpatients", linestyle="dotted", ax=ax1)

    ax1.set_ylabel("%")
    ax1.set_xlabel("Date")
    ax1.set_title("Hospital Utilization and Staffing Shortage Trends")
    ax1.legend()
    st.pyplot(fig1)

    # 2. Average Daily COVID-19 Admissions by Age Group
    st.subheader("Average Daily COVID-19 Admissions by Age Group")
    age_groups = {
        'previous_day_admission_adult_covid_confirmed_18-19': '18-19',
        'previous_day_admission_adult_covid_confirmed_20-29': '20-29',
        'previous_day_admission_adult_covid_confirmed_30-39': '30-39',
        'previous_day_admission_adult_covid_confirmed_40-49': '40-49',
        'previous_day_admission_adult_covid_confirmed_50-59': '50-59',
        'previous_day_admission_adult_covid_confirmed_60-69': '60-69',
        'previous_day_admission_adult_covid_confirmed_70-79': '70-79',
        'previous_day_admission_adult_covid_confirmed_80+': '80+'
    }
    df_age_grouped = df_grouped[list(age_groups.keys())].mean().rename(index=age_groups)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=df_age_grouped.index, y=df_age_grouped.values, palette="Blues_d", ax=ax2)
    ax2.set_ylabel("Avg. Daily Admissions")
    ax2.set_title("COVID-19 Admissions by Age Group")
    st.pyplot(fig2)

    # 3. Top 10 States by COVID-19 Admissions
    st.subheader("Top 10 States by COVID-19 Admissions")
    df_state_adm = df.groupby('state')[['previous_day_admission_adult_covid_confirmed']].sum()
    df_state_adm = df_state_adm.sort_values('previous_day_admission_adult_covid_confirmed', ascending=False).head(10)
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    df_state_adm.plot(kind='bar', ax=ax3)
    ax3.set_ylabel("Total Admissions")
    ax3.set_title("Top 10 States by COVID-19 Admissions")
    st.pyplot(fig3)

    # 4. Top 10 States by Staffing Shortages
    st.subheader("Top 10 States by Staffing Shortages")
    df_state_staff = df.groupby('state')[['critical_staffing_shortage_today_yes']].sum()
    df_state_staff = df_state_staff.sort_values('critical_staffing_shortage_today_yes', ascending=False).head(10)
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    df_state_staff.plot(kind='bar', ax=ax4, color='salmon')
    ax4.set_ylabel("Shortage Reports")
    ax4.set_title("Top 10 States by Staffing Shortages")
    st.pyplot(fig4)

    # 5. Hospital Burden & Staffing Crisis Breakdown
    st.subheader("Hospital Burden & Staffing Crisis Breakdown")
    def plot_percentage_count_plots(df, categorical_features):
        label_map = {'Red (High)': 'High', 'Yellow (Moderate)': 'Moderate', 'Green (Low)': 'Low'}
        custom_palette = {'High': 'red', 'Moderate': 'gold', 'Low': 'green'}
        fig, axes = plt.subplots(1, len(categorical_features), figsize=(6 * len(categorical_features), 5))
        if len(categorical_features) == 1:
            axes = [axes]
        for idx, feature in enumerate(categorical_features):
            df[feature] = df[feature].astype(str).str.strip().str.title()
            df[feature] = df[feature].replace(label_map)
            value_counts = df[feature].value_counts(normalize=True) * 100
            ordered = ['Low', 'Moderate', 'High']
            value_counts = value_counts.reindex([x for x in ordered if x in value_counts.index])
            sns.barplot(x=value_counts.index, y=value_counts.values,
                        palette=[custom_palette.get(x, 'gray') for x in value_counts.index], ax=axes[idx])
            axes[idx].set_title(f'{feature}')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Percentage (%)')
            for i, p in enumerate(value_counts.values):
                axes[idx].text(i, p + 1, f'{p:.1f}%', ha='center')
        plt.tight_layout()
        st.pyplot(fig)

    categorical_columns = ['Hospital_Pressure_Category', 'Staffing_Crisis_Level', 'COVID_Burden_Level']
    plot_percentage_count_plots(df, categorical_columns)

# ------------------ Inpatient Bed Utilization ------------------
elif section == "Inpatient Bed Utilization":
    st.header("🗺️ State-wise Bed Utilization")
    
    # Static Choropleth Map (Current Utilization)
    st.subheader("📍 Current State-wise Average Bed Utilization")
    state_util = df.groupby('state')['inpatient_beds_utilization'].mean().reset_index()
    state_util.columns = ['state', 'avg_utilization']

    fig_current = px.choropleth(
        state_util,
        locations='state',
        locationmode='USA-states',
        color='avg_utilization',
        color_continuous_scale='RdBu_r',
        scope='usa',
        hover_data={'state': True, 'avg_utilization': ':.2f'},
        labels={'avg_utilization': 'Avg. Utilization'}
    )
    fig_current.update_layout(
        title="Current Bed Utilization by State",
        title_font_size=20,
        height=650,
        geo=dict(
            scope="usa",
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            showland=True,
            landcolor='rgb(240,240,240)',
            subunitcolor='black',
            showframe=False,
            showcountries=False,
        ),
        coloraxis_colorbar=dict(
            title='Utilization %',
            tickformat='.2f',
            len=0.4,
            thickness=12
        )
    )
    st.plotly_chart(fig_current, use_container_width=True)

    # Prediction Map (Interactive)
    st.subheader("🔮 Predicted State-wise Bed Utilization (Interactive)")    
    weeks_ahead = st.slider("Weeks Ahead for Prediction:", min_value=1, max_value=8, value=3)
    df['year'] = 2022  
    _encoder = train_label_encoder(df)
    df['state_encoded'] = _encoder.transform(df['state'])

    features = ['state_encoded', 'week_of_year', 'year']
    df_copy = df.copy()
    df_copy = df_copy.sort_values(['state', 'week_of_year'])
    df_copy['target_util'] = df_copy.groupby('state')['inpatient_beds_utilization'].shift(-weeks_ahead)
    model_df = df_copy.dropna(subset=['target_util'])

    X = model_df[features]
    y = model_df['target_util']

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    latest_week_data = df.groupby('state').tail(1).copy()
    latest_week_data['week_of_year'] += weeks_ahead
    latest_week_data['state_encoded'] = _encoder.transform(latest_week_data['state'])
    pred_input = latest_week_data[features]
    latest_week_data['predicted_util'] = model.predict(pred_input)

    fig = px.choropleth(
        latest_week_data,
        locations='state',
        locationmode='USA-states',
        color='predicted_util',
        scope='usa',
        color_continuous_scale='RdBu_r',
        hover_data={'state': True, 'predicted_util': ':.2f'},
        labels={'predicted_util': f'Prediction (t+{weeks_ahead}w)'}
    )

    fig.update_layout(
        title=f"🛏️ Predicted State-wise Bed Utilization ({weeks_ahead} Week{'s' if weeks_ahead > 1 else ''} Ahead)",
        title_font_size=20,
        height=650,
        geo=dict(
            scope="usa",
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            showland=True,
            landcolor='rgb(240,240,240)',
            subunitcolor='black',
            showframe=False,
            showcountries=False,
        ),
        coloraxis_colorbar=dict(
            title='Utilization %',
            tickformat='.2f',
            len=0.4,
            thickness=12
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------ Staffing Shortage Section ------------------
elif section == "Staffing Shortage Status":
    st.header("👩‍⚕️ Staffing Shortage Overview")
    
    df_staff = df[["date", "state", "critical_staffing_shortage_today_yes", "critical_staffing_shortage_today_no"]].copy()
    df_staff['staff_shortage_pct'] = (
        df_staff['critical_staffing_shortage_today_yes'] /
        (df_staff['critical_staffing_shortage_today_yes'] + df_staff['critical_staffing_shortage_today_no']).replace(0, np.nan)
    ) * 100
    df_staff = df_staff.dropna()

    # ------------------ Current Average Staff Shortage ------------------
    st.subheader("📍 Current State-wise Average Staff Shortage %")
    df_current = df_staff.groupby('state')['staff_shortage_pct'].mean().reset_index()
    df_current.columns = ['state', 'avg_staff_shortage']

    fig_current = px.choropleth(
        df_current,
        locations='state',
        locationmode='USA-states',
        color='avg_staff_shortage',
        scope='usa',
        color_continuous_scale='OrRd',
        labels={'avg_staff_shortage': 'Staff Shortage %'},
        title="Current Staff Shortage by State"
    )
    fig_current.update_layout(
        title_font_size=20,
        height=650,
        geo=dict(
            scope="usa",
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            showland=True,
            landcolor='rgb(240,240,240)',
            subunitcolor='black',
            showframe=False,
            showcountries=False
        ),
        coloraxis_colorbar=dict(
            title='Shortage %',
            tickformat='.2f',
            len=0.4,
            thickness=12
        )
    )
    st.plotly_chart(fig_current, use_container_width=True)

    # ------------------ Predicted Staff Shortage ------------------
    st.subheader("🔮 Predicted Staff Shortage by State (Interactive)")
    df_weekly = df_staff.groupby(['state', pd.Grouper(key='date', freq='W-MON')])['staff_shortage_pct'].mean().reset_index()
    df_weekly = df_weekly.sort_values(by=['state', 'date'])

    for lag in [1, 2, 3]:
        df_weekly[f'lag_{lag}'] = df_weekly.groupby('state')['staff_shortage_pct'].shift(lag)

    df_weekly = df_weekly.dropna()

    X = df_weekly[['lag_1', 'lag_2', 'lag_3']]
    y = df_weekly['staff_shortage_pct']

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X, y)

    predict_weeks = st.slider("Weeks Ahead:", min_value=1, max_value=8, value=3)
    future_predictions = []

    for state in df_weekly['state'].unique():
        state_df = df_weekly[df_weekly['state'] == state].sort_values('date')
        latest_row = state_df.iloc[-1]
        lags = latest_row[['lag_1', 'lag_2', 'lag_3']].values

        for step in range(predict_weeks):
            next_pred = model_rf.predict(pd.DataFrame([lags], columns=['lag_1', 'lag_2', 'lag_3']))[0]
            future_predictions.append({
                'state': state,
                'week': step + 1,
                'predicted_shortage': next_pred
            })
            lags = [next_pred, lags[0], lags[1]]

    pred_df = pd.DataFrame(future_predictions)
    map_df = pred_df[pred_df['week'] == predict_weeks]

    fig_staff = px.choropleth(
        map_df,
        locations='state',
        locationmode='USA-states',
        color='predicted_shortage',
        scope='usa',
        color_continuous_scale='OrRd',
        hover_name='state',
        labels={'predicted_shortage': 'Staff Shortage %'},
        title=f"💉Predicted Staff Shortage - {predict_weeks} Week{'s' if predict_weeks > 1 else ''} Ahead"
    )
    fig_staff.update_layout(
        title_font_size=20,
        height=650,
        geo=dict(
            scope="usa",
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            showland=True,
            landcolor='rgb(240,240,240)',
            subunitcolor='black',
            showframe=False,
            showcountries=False
        ),
        coloraxis_colorbar=dict(
            title='Shortage %',
            tickformat='.2f',
            len=0.4,
            thickness=12
        )
    )
    st.plotly_chart(fig_staff, use_container_width=True)
    
# ------------------ Scenario Planning Section ------------------
elif section == "Scenario Planning":
    st.header("💡 Scenario Planning")
    st.markdown("""
> ℹ️ **Note:** This scenario module is designed for exploratory purposes.
> Predictions are based on COVID-period data, so changing everyday conditions may not significantly affect the results.
""")

    st.markdown("### Conditions")
    seasons = st.multiselect("Seasons", ["Spring", "Summer", "Fall", "Winter"])
    admission_rate = st.slider("Admission Rate", 0, 50, 15)
    emergency_surge = st.radio("Emergency Surge", ["No Surge", "+10%", "+20%", "+30%"])
    staff_current = st.radio("Staff Current Shortage", ["Yes", "No"])
    staff_anticipated = st.radio("Staff Anticipated Shortage", ["Yes", "No"])
    available_beds = st.slider("Available Beds (%)", -25, 25, 0, format="%d%%")
    confirmed = st.button("Confirm")

    if confirmed:
        _encoder = train_label_encoder(df)
        model = train_bed_utilization_model(df, _encoder)
        staff_model = train_staff_shortage_model(df, _encoder)

        latest_week = df['week_of_year'].max()
        year = 2022
        factor_map = {"+10%": 0.1, "+20%": 0.2, "+30%": 0.3, "No Surge": 0.0}
        emergency_surge = factor_map.get(emergency_surge, 0.0)

        all_states = df['state'].unique()
        input_data = pd.DataFrame({
            'state': all_states,
            'week_of_year': latest_week,
            'year': year,
            'admission_rate': admission_rate,
            'emergency_surge': emergency_surge,
            'staff_current_shortage': 1 if staff_current == "Yes" else 0,
            'staff_anticipated_shortage': 1 if staff_anticipated == "Yes" else 0,
            'available_beds': available_beds
        })

        input_data['state_encoded'] = _encoder.transform(input_data['state'])
        features = ['state_encoded', 'week_of_year', 'year', 'admission_rate',
                    'emergency_surge', 'staff_current_shortage',
                    'staff_anticipated_shortage', 'available_beds']

        X_input = input_data[features].astype(float)
        input_data['predicted_utilization'] = model.predict(X_input)

        # Plot
        fig_scenario = px.choropleth(
            input_data,
            locations='state',
            locationmode='USA-states',
            color='predicted_utilization',
            scope='usa',
            color_continuous_scale='RdBu_r',
            hover_data={'state': True, 'predicted_utilization': ':.2f'},
            labels={'predicted_utilization': 'Utilization (%)'},
            title="🛏️ Predicted Bed Utilization by Scenario"
        )

        fig_scenario.update_layout(
            title_font_size=20,
            height=650,
            geo=dict(
                scope="usa",
                showlakes=True,
                lakecolor='rgb(255, 255, 255)',
                showland=True,
                landcolor='rgb(240,240,240)',
                subunitcolor='black',
                showframe=False,
                showcountries=False
            ),
            coloraxis_colorbar=dict(
                title='Utilization %',
                tickformat='.2f',
                len=0.4,
                thickness=12
            )
        )
        st.plotly_chart(fig_scenario, use_container_width=True)

        # Load staff shortage model
        input_data['predicted_staffing_shortage'] = staff_model.predict(X_input)

        # Staffing shortage choropleth
        fig_staff_scenario = px.choropleth(
            input_data,
            locations='state',
            locationmode='USA-states',
            color='predicted_staffing_shortage',
            scope='usa',
            color_continuous_scale='OrRd',
            hover_data={'state': True, 'predicted_staffing_shortage': ':.2f'},
            labels={'predicted_staffing_shortage': 'Staff Shortage (%)'},
            title="👩‍⚕️ Predicted Staffing Shortage by Scenario"
        )

        fig_staff_scenario.update_layout(
            title_font_size=20,
            height=650,
            geo=dict(
                scope="usa",
                showlakes=True,
                lakecolor='rgb(255, 255, 255)',
                showland=True,
                landcolor='rgb(240,240,240)',
                subunitcolor='black',
                showframe=False,
                showcountries=False
            ),
            coloraxis_colorbar=dict(
                title='Shortage %',
                tickformat='.2f',
                len=0.4,
                thickness=12
            )
        )

        st.plotly_chart(fig_staff_scenario, use_container_width=True)

