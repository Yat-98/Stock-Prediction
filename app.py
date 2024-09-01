from flask import Flask, render_template, Response
import pandas as pd
import plotly.express as px
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import io
import plotly.io as pio
import base64

app = Flask(__name__)
pio.renderers.default = 'json'  # Change as needed for your environment

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/plot')
def plot():
    # Load the CSV file
    file_path = 'TSLA.csv'
    df = pd.read_csv(file_path)

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Preparing data for Prophet
    columns = ['Date', "Close"]
    ndf = pd.DataFrame(df, columns=columns)
    prophet_df = ndf.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Fitting the model
    m = Prophet()
    m.fit(prophet_df)

    # Making future predictions
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    # Plotting using Plotly
    fig1 = px.line(df, x="Date", y="Close", title='Historical Close Prices')
    fig2 = px.line(forecast, x='ds', y='yhat', title='Forecasted Prices')
    
    # Save Plotly figures to HTML
    fig1_html = pio.to_html(fig1, full_html=False)
    fig2_html = pio.to_html(fig2, full_html=False)

    # Matplotlib plots
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['Date'], df['Close'], label='Close', color='blue')
    ax.plot(forecast['ds'][:len(df)], forecast['yhat'][:len(df)], label='yhat', color='red')
    ax.set_title('Evaluation')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    plt.close(fig)

    # Save Matplotlib figure to BytesIO object
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    
    # Encode the image to base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template('plot.html', plot1=fig1_html, plot2=fig2_html, matplot_img=img_base64)

if __name__ == '__main__':
    app.run(debug=True)

