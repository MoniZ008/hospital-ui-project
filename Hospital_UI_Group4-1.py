import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Hospital Bed and Resource Forecasting", layout="wide")
st.image("uic.png", use_container_width=True)

# ------------------ Load Data ------------------
data_path = "/Users/monica/Desktop/MSBA/2025 Spring/IDS560/project-group4/weekly tasks/week14-0419/ui/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW__20250220.csv"
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week
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

# ------------------ Navigation ------------------
section = st.sidebar.radio("Navigation", ["Overview","Inpatient Bed Utilization", "Staffing Shortage Status", "Scenario Planning"])

# ------------------ Overview ------------------
if section == "Overview":
    st.header("📊 Overview")

    # 1. Hospital Bed Utilization Over Time
    st.subheader("Hospital Bed Utilization Over Time")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df, x="date", y="inpatient_beds_utilization", label="Inpatient Bed Utilization")
    sns.lineplot(data=df, x="date", y="staffing_shortage_ratio", label="Staffing Shortage Ratio", linestyle="dashed")
    sns.lineplot(data=df, x="date", y="percent_of_inpatients_with_covid", label="% COVID Inpatients", linestyle="dotted")
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
    df['year'] = 2022  # placeholder year for modeling
    le = LabelEncoder()
    df['state_encoded'] = le.fit_transform(df['state'])

    features = ['state_encoded', 'week_of_year', 'year']
    df_copy = df.copy()
    df_copy = df_copy.sort_values(['state', 'week_of_year'])
    df_copy['target_util'] = df_copy.groupby('state')['inpatient_beds_utilization'].shift(-weeks_ahead)
    model_df = df_copy.dropna(subset=['target_util'])

    X = model_df[features]
    y = model_df['target_util']

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    latest_week_data = df.groupby('state').tail(1).copy()
    latest_week_data['week_of_year'] += weeks_ahead
    latest_week_data['state_encoded'] = le.transform(latest_week_data['state'])
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
    st.header("👩‍⚕️ Staffing Availability Overview")

    df_staff = df[["date", "state", "critical_staffing_shortage_today_yes", "critical_staffing_shortage_today_no"]].copy()
    df_staff['Staff Availability %'] = (
        df_staff['critical_staffing_shortage_today_no'] /
        (df_staff['critical_staffing_shortage_today_yes'] + df_staff['critical_staffing_shortage_today_no']).replace(0, np.nan)
    ) * 100
    df_staff = df_staff.dropna()

    # ------------------ Current Average Staff Availability ------------------
    st.subheader("📍 Current State-wise Average Staff Availability")
    df_current = df_staff.groupby('state')['Staff Availability %'].mean().reset_index()

    fig_current = px.choropleth(
        df_current,
        locations='state',
        locationmode='USA-states',
        color='Staff Availability %',
        scope='usa',
        color_continuous_scale='PuBu',
        title=f"Current Staff Availability by State"
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
            title='Staff Availability %',
            tickformat='.2f',
            len=0.4,
            thickness=12
        )
    )
    st.plotly_chart(fig_current, use_container_width=True)

    # ------------------ Predicted Staff Availability ------------------
    st.subheader("🔮 Predicted Staff Availability by State (Interactive)")

    df_weekly = df_staff.groupby(['state', pd.Grouper(key='date', freq='W-MON')])['Staff Availability %'].mean().reset_index()
    df_weekly = df_weekly.sort_values(by=['state', 'date'])

    for lag in [1, 2, 3]:
        df_weekly[f'lag_{lag}'] = df_weekly.groupby('state')['Staff Availability %'].shift(lag)

    df_weekly = df_weekly.dropna()

    X = df_weekly[['lag_1', 'lag_2', 'lag_3']]
    y = df_weekly['Staff Availability %']

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
                'Predicted Staff Availability %': next_pred
            })
            lags = [next_pred, lags[0], lags[1]]

    pred_df = pd.DataFrame(future_predictions)
    map_df = pred_df[pred_df['week'] == predict_weeks]


    fig_staff = px.choropleth(
        map_df,
        locations='state',
        locationmode='USA-states',
        color='Predicted Staff Availability %',
        scope='usa',
        color_continuous_scale='PuBu',
        hover_name='state',
        title=f"🩺 Predicted Staff Availability - {predict_weeks} Week{'s' if predict_weeks > 1 else ''} Ahead"
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
            title='Staff Availability %',
            tickformat='.2f',
            len=0.4,
            thickness=12
        )
    )
    st.plotly_chart(fig_staff, use_container_width=True)
    
# ------------------ Scenario Planning Section ------------------
elif section == "Scenario Planning":
    st.header("💡 Scenario Planning")

    st.markdown("### Conditions")
    seasons = st.multiselect("Seasons", ["Spring", "Summer", "Fall", "Winter"])
    admission_rate = st.slider("Admission Rate", 0, 50, 15)
    emergency_surge = st.radio("Emergency Surge", ["No Surge", "+10%", "+20%", "+30%"])
    staff_current = st.radio("Staff Current Shortage", ["Yes", "No"])
    staff_anticipated = st.radio("Staff Anticipated Shortage", ["Yes", "No"])
    available_beds = st.slider("Available Beds", -25, 25, 0)
    confirmed = st.button("Confirm")

    if confirmed:
        sim_df = df.groupby('date')[['inpatient_beds_utilization']].mean().reset_index()
        factor = 1 + admission_rate / 100 + (0.1 if "+10%" in emergency_surge else 0.2 if "+20%" in emergency_surge else 0.3 if "+30%" in emergency_surge else 0)
        sim_df['adjusted_utilization'] = sim_df['inpatient_beds_utilization'] * factor
        fig_adj = px.line(sim_df, x='date', y=['inpatient_beds_utilization', 'adjusted_utilization'],
                          labels={'value': 'Utilization'}, title="Adjusted vs Original Inpatient Bed Utilization")
        st.plotly_chart(fig_adj, use_container_width=True)
