from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime

import threading
import time

# Set period under observation
start_date = datetime.datetime(2022, 7, 1)
end_date = datetime.datetime(2024, 5, 31)

# Sentiment data preprocessing
# read twitter sentiment data 
tweets_sentiment = pd.read_csv('../Data/tweets_sentiment.csv', index_col='Unnamed: 0',  lineterminator='\n')

# compute tweet sentiment score and label
tweets_sentiment['sentiment_score'] = tweets_sentiment['Positive'] - tweets_sentiment['Negative']

# read twitter topic data 
tweets_topic = pd.read_csv('../Data/tweets_topic.csv', index_col='Unnamed: 0')

# merge twitter sentiment and topic into one dataframe
sentiment_df = tweets_sentiment.merge(tweets_topic, how='inner', on='id', suffixes=('', '_copy'))
sentiment_df.drop(columns=[col for col in sentiment_df.columns if col.endswith('_copy')], inplace=True)

# filter for tweets within date range
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
sentiment_df = sentiment_df[(sentiment_df['date'] >= start_date) & (sentiment_df['date'] <= end_date)]

# Group tweets by date and calculate average sentiment
daily_sentiment = sentiment_df.loc[:, ['date', 'sentiment_score']]
daily_sentiment = daily_sentiment.groupby(['date']).mean()
daily_sentiment.columns = [f'{col}_0' for col in daily_sentiment.columns]

# Group by date and topic and calculate average sentiment
daily_topic_sentiment = sentiment_df.loc[:, ['date', 'topic', 'sentiment_score']]
daily_topic_sentiment['topic'] = daily_topic_sentiment['topic']+1
daily_topic_sentiment = daily_topic_sentiment.groupby(['date', 'topic']).mean()

# Unstack 'topic' to become columns
daily_topic_sentiment = daily_topic_sentiment.unstack(level='topic')

# Flatten the column MultiIndex resulting from unstack
daily_topic_sentiment.columns = daily_topic_sentiment.columns.map('{0[0]}_{0[1]}'.format) 

# merge daily sentiment by topic and overall into one dataframe
daily_sentiment = pd.concat([daily_sentiment, daily_topic_sentiment], axis=1)
daily_sentiment = pd.melt(daily_sentiment, ignore_index=False)
daily_sentiment.reset_index(inplace=True)

# # Mock data for returns
# # Set a random seed for reproducibility
# np.random.seed(42)

# # Generate a mock date range based on start and end dates
# dates = pd.date_range(start=start_date, end=end_date, freq='D')

# # Generate random predicted returns and actual returns for the date range
# predicted_returns = np.random.normal(0, 0.02, len(dates))  # Mean 0, Std 0.02
# actual_returns = np.random.normal(0, 0.02, len(dates))  # Mean 0, Std 0.02

# # Simulate positions: 1 for buy, -1 for sell, based on predicted returns
# positions = np.where(predicted_returns > 0, 1, -1)

# # Calculate strategy returns based on positions
# strategy_returns = positions * actual_returns

# # Calculate cumulative returns (actual and strategy)
# cumulative_actual_returns = (1 + actual_returns).cumprod()
# cumulative_strategy_returns = (1 + strategy_returns).cumprod()

# # Create the DataFrame
# strategy_df = pd.DataFrame({
#     'date': dates,
#     'predicted_return': predicted_returns,
#     'actual_return': actual_returns,
#     'position': positions,
#     'strategy_return': strategy_returns,
#     'cumulative_actual_return': cumulative_actual_returns,
#     'cumulative_strategy_return': cumulative_strategy_returns
# })
# Get predicted return of the models
predictions_df = pd.read_csv('../Data/predictions.csv')
predictions_lstm_df = pd.read_csv('../Data/predictions_lstm.csv')
predictions_df = pd.merge(predictions_df, predictions_lstm_df, on=['date', 'Actual'], suffixes=['', '_LSTM'])

# Convert returnn to buy/sell signals
positions_df = predictions_df.copy()
for col in positions_df.columns[2:]:
    if col.startswith('train_set'):
        continue
    positions_df[col] = positions_df[col].apply(lambda x: 1 if x>0 else -1)

prices = pd.read_csv('../Data/DCOILBRENTEU.csv')
prices.rename(columns={'DATE':'date', 'DCOILBRENTEU':'close'}, inplace=True)

prices = pd.merge(prices, positions_df, on='date')
prices.drop(columns=['Actual', 'train_set', 'train_set_LSTM'], inplace=True)

prices[prices.columns[3:]] = prices[prices.columns[3:]].shift(-1)

prices['date'] = pd.to_datetime(prices['date'])
prices.set_index('date', inplace=True)

latest_brent_oil_price = pd.DataFrame()
# Function to continuously fetch data
def fetch_brent_oil_price():
    global latest_brent_oil_price
    while True:
        try:
            latest_brent_oil_price = web.DataReader('DCOILBRENTEU', 'fred', start_date, datetime.datetime.today())
            print(f"Updated data: {latest_brent_oil_price.tail()}")
        except Exception as e:
            print(f"Error fetching data: {e}")
        time.sleep(60)  # Update every 60 seconds

# Start the data fetching thread
data_fetching_thread = threading.Thread(target=fetch_brent_oil_price, daemon=True)
data_fetching_thread.start()

def register_callbacks(app):
    @app.callback(
        Output('sentiment-graph', 'figure'),
        [Input('topic-dropdown', 'value')]
    )
    def update_sentiment_chart(topic):
        filtered_df = daily_sentiment 
        fig = px.line(filtered_df, x="date", y="value", color='variable')
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        return fig

    # @app.callback(
    #     Output('cumulative-return-chart', 'figure'),
    #     [Input('threshold-slider', 'value')]
    # )
    # def update_cumulative_return(threshold):
    #     strategy_df['position'] = strategy_df['predicted_return'].apply(lambda x: 1 if x > threshold else -1)
    #     strategy_df['strategy_return'] = strategy_df['position'] * strategy_df['actual_return']
    #     strategy_df['cumulative_strategy_return'] = (1 + strategy_df['strategy_return']).cumprod()
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=strategy_df['date'], y=strategy_df['cumulative_actual_return'], mode='lines', name='Actual Cumulative Return'))
    #     fig.add_trace(go.Scatter(x=strategy_df['date'], y=strategy_df['cumulative_strategy_return'], mode='lines', name='Strategy Cumulative Return'))
    #     return fig
    
    @app.callback(
        Output("cumulative-return-chart", "figure"),
        Input("run-backtest", "n_clicks"),
        State("model-dropdown", "value")
    )
    def update_backtest(n_clicks, selected_model):
        if n_clicks == 0:
            return go.Figure()

        results=prices[['close', selected_model]]
        results.rename(columns={selected_model:'signal'}, inplace=True)

        # setup counting system
        holding=0        
        holdingcost=0    
        outflow=0        
        inflow=0         
        order={}

        for date,row in results.iterrows():

            # if signal is openpos(signal=1), and there's no holding bitcoin(holding=0) then
            if row['signal']==1 and holding==0:

                # add 1 bitcoin into holding
                holding=holding+1
                # track on all-time cash outflow
                outflow-=row['close']
                # track on the single time buyin cost
                holdingcost+=row['close']
                # recortd the cost and buytime into dict
                order[date]=-row['close']
                # print out the on-time buyin action
                print('Buy:{buy_create}$ @ {dt}'.format(buy_create=order[date],dt=date))

            # else if signal is less than 0,and holding any of bitcoin then sell out all
            elif row['signal']< 0 and holding>0:

                # track the all-time cash inflow (before selling)
                inflow+= row['close']*holding
                # record the sell action into dict             
                order[date]=+row['close']*holding
                # reset the holding number (since we have sold all)
                holding=0
                # reset the holding cost (since we don't hold anything now)
                holdingcost=0
                # print out the on-time sell out action
                print('Sell:{sell_create}$ @ {dt}'.format(sell_create=round(order[date],2),dt=date)) ## round

            # record bitcoin holding qty in dataframe 
            results.loc[date,'holding']=holding
            # record currently portfolio value in dataframe
            results.loc[date,'portvalue']=results.loc[date,'holding']*row['close']
            # record currently holding cost in dataframe
            results.loc[date,'holdingcost']=holdingcost

            # record profit in each timestamp
            results.loc[date,'profit']=inflow+outflow+results.loc[date,'portvalue']

        # calculation of the ROI
        results['roi']=(results['portvalue']-results['holdingcost'])/results['holdingcost']
        results['roi']=results['roi'].fillna(0)

        # Cumulative Return Chart
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=results.index, y=results['profit'], mode='lines', name='Cumulative Return'))
        # fig.update_layout(title="Cumulative Return Over Time", xaxis_title="Date", yaxis_title="Cumulative Return")
        
        # Cumulative Return Chart with Signals
        # create empty list for saving buy_date and sell_date
        buy_date=[]
        sell_date=[]

        # loop over the order dictionary and append buy & sell date
        for key,value in order.items():
            if value>0:
                sell_date.append(key)
            if value<0:
                buy_date.append(key)

        # extract the buy price and sell price
        sp=results.loc[sell_date,'close']
        bp=results.loc[buy_date,'close']
        
        # setup plotly object figure
        fig=go.Figure()

        # adding trace of close price and profit
        fig.add_trace(go.Scatter(x=results.index,y=results['close'],name='Close',line_color='SlateBlue'))

        fig.add_trace(go.Scatter(x=results.index,y=results['profit'],name='Profit',line_color='LightPink'))

        # adding marker of Buyin and sell out 
        fig.add_trace(go.Scatter(x=buy_date,y=np.abs(bp),
                                name='Buy',mode='markers',
                                marker=dict(size=10,symbol=5),
                                marker_color='red',
                                text='BUY'))

        fig.add_trace(go.Scatter(x=sell_date,y=np.abs(sp),
                                name='Sell',mode='markers',
                                marker=dict(size=10,symbol=6),
                                marker_color='lime',
                                text='SELL'))

        # styling the plot and put text on it
        fig.update_layout(title_text=f'Backtest:{selected_model}',
                        xaxis_rangeslider_visible=True,
                        template='gridon')

        return fig
                     
    @app.callback(
    Output('brent-price-chart', 'figure'),
    [Input('threshold-slider', 'value')]
    )
    def update_brent_price(threshold):
        # Get price and returns
        price_df = pd.read_csv('../Data/DCOILBRENTEU.csv')
        price_df['DATE'] = pd.to_datetime(price_df['DATE'])
        price_df = price_df[(price_df['DATE'] > start_date) & (price_df['DATE'] < end_date)]

        price_df.columns=['date', 'price']
        price_df['returns'] = price_df['price'].pct_change(1, fill_method=None)
        price_df.head()

        # Create the figure object
        fig = go.Figure()

        # Add price trace on the left y-axis
        fig.add_trace(go.Scatter(x=price_df['date'], y=price_df['price'], 
                                mode='lines', name='Price'))

        # Add returns trace on the right y-axis
        fig.add_trace(go.Scatter(x=price_df['date'], y=price_df['returns'], 
                                mode='lines', name='Returns', yaxis='y2'))

        # Create a secondary y-axis for returns
        fig.update_layout(
            title='Price and Returns Over Time',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Returns', overlaying='y', side='right')
        )

        return fig

    @app.callback(
    Output('live-brent-price-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
    )
    def update_live_brent_price(n):
        global latest_brent_oil_price

        if not latest_brent_oil_price.empty:
            fig = {
                'data': [{
                    'x': latest_brent_oil_price.index,
                    'y': latest_brent_oil_price['DCOILBRENTEU'],
                    'type': 'line'
                }],
                'layout': {
                    'title': 'Live Brent Oil Prices'
                }
            }
        else:
            # Provide an empty figure or an error message if no data is available
            fig = {
                'data': [],
                'layout': {
                    'title': 'Live Brent Oil Prices (No Data Available)'
                }
            }
        return fig


    
