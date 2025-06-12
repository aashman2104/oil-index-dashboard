import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import numpy as np

st.set_page_config(page_title="Custom Oil Index Dashboard", layout="wide")
st.title("🔃 Custom Oil Index Dashboard")

@st.cache_data
def load_data():
    prices = pd.read_csv("merged_prices.csv")# Raw price data for 6 oil components
    index = pd.read_csv("oil_index.csv") # Pre-existing index (not used in calculation)

    prices["Date"] = pd.to_datetime(prices["Date"], format='mixed', dayfirst=True, errors='coerce')
    index["Date"] = pd.to_datetime(index["Date"], format='mixed', dayfirst=True, errors='coerce')

    prices.set_index("Date", inplace=True)
    index.set_index("Date", inplace=True)

    return prices, index

merged_df, index_df = load_data()
returns_df = merged_df.pct_change().dropna() # calculates daily percentage returns - creates dataframe with each column for each component

# --- Inverse Volatility Weighted Index ---
rolling_vol = returns_df.rolling(30).std() #30 day rolling std deviation
index_returns = []

for date, row in returns_df.iterrows():
    if date not in rolling_vol.index:
        continue
    vols = rolling_vol.loc[date].dropna() #get volatilities for this date
    inv_vols = (1 / vols)  # calculate inverse volatilities ( lower vol = higher weight)
    weights = inv_vols / inv_vols.sum() #normalize to weights that sum to 1 - converts raw inverse volatility "scores" into proper portfolio weights that sum to 1 (or 100%)
    valid_cols = [col for col in weights.index if col in row.index] #ensures we only use components with valid data 
    weighted_return = (row[valid_cols] * weights[valid_cols]).sum() #calculate weighted returns for the day for each oil and sums it
    index_returns.append((date, weighted_return)) #tags the weighted returns for all 6 oil (comibined) to the respective date

index_returns_series = pd.Series(dict(index_returns)).sort_index() #creates an index to store and sort the returns 
oil_index_cum = 100 * (1 + index_returns_series).cumprod() #converts daily returns to cumulative index starting at 100 and compounds the returns
index_df = pd.DataFrame({"Oil Index": oil_index_cum})

# --- Line Chart ---
st.subheader("📈 Index Performance")
st.line_chart(index_df)

# --- Return Summary Panel ---
st.subheader("📊 Index Return Summary")

def calc_return(df, days):
    if len(df) < days:
        return None
    col = df.columns[0]
    current = df[col].iloc[-1]
    past = df[col].iloc[-days]
    return (current / past - 1) * 100

try:
    returns = {
        "1D": calc_return(index_df, 2),
        "1W": calc_return(index_df, 6),
        "1M": calc_return(index_df, 22),
        "YTD": calc_return(
            index_df[index_df.index >= pd.Timestamp(f"{pd.Timestamp.today().year}-01-01")],
            len(index_df[index_df.index >= pd.Timestamp(f"{pd.Timestamp.today().year}-01-01")])
        )
    }
    returns_df_summary = pd.DataFrame.from_dict(returns, orient='index', columns=['Return (%)'])
    st.dataframe(returns_df_summary.style.format({"Return (%)": "{:+.2f}"}))
except Exception as e:
    st.warning(f"Could not calculate returns: {e}")

# --- Component Snapshot ---
st.subheader("🧪 Component Prices (Latest)")
latest_prices = merged_df.iloc[-1]
previous_prices = merged_df.iloc[-2]
price_change = (latest_prices - previous_prices) / previous_prices * 100

component_df = pd.DataFrame({
    "Latest Price": latest_prices,
    "Daily Change (%)": price_change.round(2)
})
st.dataframe(component_df.style.format({"Latest Price": "{:.2f}", "Daily Change (%)": "{:+.2f}"}))

# --- Weights Display ---
st.subheader("⚖️ Index Weights")
weights_df_display = pd.DataFrame({
    "Oil": ['Palm Oil', 'Soybean Oil', 'Rapeseed Oil', 'Gas Oil', 'Brent Crude', 'WTI Crude'],
    "Weight (%)": ["Dynamic by Volatility"] * 6
})
st.dataframe(weights_df_display)

# --- Correlation with Index ---
st.subheader("📌 Correlation with Index")
combined_returns = returns_df.copy()
combined_returns["Oil Index"] = index_returns_series
correlations = combined_returns.corr()["Oil Index"].drop("Oil Index").sort_values(ascending=False)
correlation_df = pd.DataFrame(correlations).rename(columns={"Oil Index": "Correlation"})
st.dataframe(correlation_df.style.format({"Correlation": "{:.2f}"}))

# --- Movement Breakdown ---
st.subheader("🔍 Daily Movement Breakdown")
index_return_today = index_returns_series.iloc[-1]
index_direction = "up" if index_return_today > 0 else "down"
component_returns_today = returns_df.iloc[-1]
same_direction_mask = (component_returns_today > 0) == (index_return_today > 0)
same_direction_count = same_direction_mask.sum()
same_direction_oils = component_returns_today[same_direction_mask].index.tolist()

st.write(f"Today the index moved **{index_direction} {index_return_today:.2%}**.")
st.write(f"{same_direction_count} out of 6 components moved in the same direction as the index.")
st.write("These components:", ', '.join(same_direction_oils))


# --- Rolling Correlation with Index ---
st.subheader("🔄 30-Day Rolling Correlation with Index")

rolling_corr_df = pd.DataFrame(index=returns_df.index)

for col in returns_df.columns:
    rolling_corr_df[col] = returns_df[col].rolling(30).corr(index_returns_series)

# Plot
fig_corr = go.Figure()
for col in rolling_corr_df.columns:
    fig_corr.add_trace(go.Scatter(
        x=rolling_corr_df.index,
        y=rolling_corr_df[col],
        mode='lines',
        name=col
    ))

fig_corr.update_layout(
    title="30-Day Rolling Correlation with Index",
    xaxis_title="Date",
    yaxis_title="Correlation",
    template="plotly_white",
    height=500
)
st.plotly_chart(fig_corr, use_container_width=True)

# --- Component Directional Accuracy vs Index ---

# Align returns
aligned_returns = returns_df.loc[index_returns_series.index]

# Compute sign comparison
accuracy_df = (np.sign(aligned_returns) == np.sign(index_returns_series.to_numpy()[:, np.newaxis])).astype(int)


# --- Directional Accuracy Summary Table ---
st.subheader("📊 Component–Index Direction Match Count")

summary_counts = pd.DataFrame({
    "Total Days": accuracy_df.count(),
    "Same Direction Days": accuracy_df.sum()
})
summary_counts["Accuracy (%)"] = (summary_counts["Same Direction Days"] / summary_counts["Total Days"]) * 100
st.dataframe(summary_counts.style.format({"Accuracy (%)": "{:.2f}"}))


# --- Oil Index vs Components (Return-Based Comparison) ---
st.subheader("📉 Oil Index vs Components (Return-Based)")

component_cum_returns = (1 + returns_df).cumprod() * 100 
aligned_index = oil_index_cum.index.intersection(component_cum_returns.index)
plot_df = component_cum_returns.loc[aligned_index].copy()
plot_df["Oil Index"] = oil_index_cum.loc[aligned_index]

selected = st.multiselect(
    "Select series to compare:",
    options=plot_df.columns.tolist(),
    default=["Oil Index"]
)

if selected:
    fig = go.Figure()
    color_map = {
        "Palm Oil": "orange",
        "Soybean Oil": "green",
        "Rapeseed Oil": "purple",
        "Gas Oil": "brown",
        "Brent Crude": "blue",
        "WTI Crude": "gray",
        "Oil Index": "black"
    }
    for col in selected:
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df[col],
            mode='lines',
            name=col,
            line=dict(color=color_map.get(col, None))
        ))

    fig.update_layout(
        title="Cumulative Return Comparison (Base 100)",
        xaxis_title="Date",
        yaxis_title="Value (Base 100)",
        template="plotly_white",
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

    #######
    st.subheader("🧪 Final Day Dynamic Weights")
final_day_vols = rolling_vol.dropna().iloc[-1]
inv_vols = 1 / final_day_vols
final_weights = inv_vols / inv_vols.sum()
st.dataframe(final_weights.rename("Weight (%)").to_frame().style.format("{:.2%}"))

st.subheader("📊 20-Day Rolling Volatility")

vol_plot = rolling_vol.copy()
vol_plot = vol_plot.loc[oil_index_cum.index]  # align with index timeframe

fig_vol = go.Figure()
for col in vol_plot.columns:
    fig_vol.add_trace(go.Scatter(
        x=vol_plot.index,
        y=vol_plot[col],
        mode='lines',
        name=col
    ))

fig_vol.update_layout(
    title="Component 20-Day Rolling Volatility",
    xaxis_title="Date",
    yaxis_title="Volatility (Std Dev)",
    template="plotly_white",
    height=500
)
st.plotly_chart(fig_vol, use_container_width=True)


