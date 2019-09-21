import dash
import dash_core_components as dcc
import dash_html_components as html
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objs as go

app = dash.Dash()
server = app.server

name = pd.read_csv('NTAname.csv')
df_true = pd.read_csv('data_model_nta.csv')
df_true['fi population '] = df_true['fi population '] * 100 / df_true['Total population']
df_2 = pd.read_csv('final_df_4.csv')

df = pd.read_csv('data_model_nta.csv', delimiter=',', index_col=0)
df['fi population '] = df['fi population ']/df['Total population']

X = df.iloc[:, 1:6]
Y = df.iloc[:, 6:10]
scaler = MinMaxScaler()
scaler.fit(X)
X_norm = scaler.transform(X)
scaler2 = MinMaxScaler()
scaler2.fit(Y)
Y_norm = scaler2.transform(Y)
filename = 'finalized_model3.sav'
model = pickle.load(open(filename, 'rb'))

available_indicators = name['name'].unique()
code_default = 'MN22'

colors = {
    'background': '#efcd17',
    'text': 'black',
    'test2': '#20639B',
    'b': '#1530DB',
    'g': '#14A76C',
    'r': '#C3073F'}


def updateresult(per, un, high, college, us):
    inputs = np.array([per/100, un, high, college, us])
    inputs = inputs.reshape(1,-1)
    inputs_1 = scaler.transform(inputs)
    outputs = model.predict(inputs_1)
    outputs_1 = scaler2.inverse_transform(outputs)
    #print(outputs_1)
    return outputs_1


def update_graph_1_1(code1):
    code = name[(name['NTAcode'] == code1)]['name']
    df_nta_1 = df_true[(df_true['code'] == code1.split()[0])]
    return {
        'data': [
            {'x': [df_nta_1.iloc[0, 6],
                   df_nta_1.iloc[0, 5],
                   df_nta_1.iloc[0, 4],
                   df_nta_1.iloc[0, 3],
                   df_nta_1.iloc[0, 2] * 100],
             'y': ['Percentage of none US Citizen', 'Percent with college degree', 'percent with high school degree',
                   'Unemployment Rate', 'Percent house hold living under poverty line'],
             'type': 'bar', 'orientation': 'h', 'name': str(code.iloc[0]), 'marker': dict(color=colors['b'])},
        ],
        'layout': go.Layout(
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 1, 'y': 1},
            hovermode='closest',
            title='About ' + code.iloc[0],
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            xaxis=dict(title='Hover over the bar see attribute !',
                       titlefont=dict(size=20)),
            yaxis=dict(tickfont=dict(size=1))
        )
    }


def update_graph_1_2(code1):
    code = name[(name['NTAcode'] == code1)]['name']
    df_nta_1 = df_true[(df_true['code'] == code1.split()[0])]
    return {
        'data': [
            {'x': [str(code1.split()[0])], 'y': [df_nta_1.iloc[0, 7]], 'type': 'bar',
             'name': str(code1.split()[0]), 'marker': dict(color=colors['b'])}],
        'layout': go.Layout(
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            yaxis={'range': [0, 40]},
            legend=dict(x=1, y=1, font=dict(family='sans-serif', size=1)),
            hovermode='closest', title='Food Insecurity Population %',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
        )
    }


def update_graph_1_3(code1):
    code = name[(name['NTAcode'] == code1)]['name']
    df_nta_1 = df_true[(df_true['code'] == code1.split()[0])]
    return {
        'data': [
            {'x': [str(code1.split()[0])], 'y': [df_nta_1.iloc[0, 8]], 'type': 'bar',
             'name': str(code1.split()[0]), 'marker': dict(color=colors['b'])}],
        'layout': go.Layout(
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            yaxis={'range': [0, 2500000]},
            legend=dict(x=1, y=1, font=dict(family='sans-serif', size=1)),
            hovermode='closest', title='Meal Gap',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
        )
    }


def update_graph_1_4(code1):
    code = name[(name['NTAcode'] == code1)]['name']
    df_nta_1 = df_true[(df_true['code'] == code1.split()[0])]
    return {
        'data': [
            {'x': [str(code1.split()[0])], 'y': [df_nta_1.iloc[0, 9]], 'marker': dict(color=colors['b']),
             'type': 'bar', 'name': str(code1.split()[0])}],
        'layout': go.Layout(
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            yaxis={'range': [0, 1500000]},
            legend=dict(x=1, y=1, font=dict(family='sans-serif', size=1)),
            hovermode='closest', title='Supply Gap',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
        )
    }


app.layout = html.Div([
    # Title Main Page
    html.Div([
        html.Div([
            html.H1('   ')
        ], style={'width': '100%', 'color': colors['text'], 'display': 'block', 'font-size': 40,
                  'height': '200px'}),
        html.Div([
            html.H1('Design Your Own City ! ')
        ], style={'width': '100%', 'color': colors['text'], 'display': 'block', 'font-size': 40,
                  'height': '200px'}),
        html.Div([
            html.P('With City Harvest\'s Data')
        ], style={'width': '100%', 'color': colors['text'], 'display': 'block', 'font-size': 30}),
        html.Div([
            html.P('City Harvest is New York City\'s largest food rescue organization feeding more than 1.2 million '
                   'New Yrkers who are food insecure. The organization, in 2018, collected 61 million pounds of food '
                   'from restaurants, groceries, and farms that had an excess supply and distributed it to New Yorkers '
                   'struggling to put food on their tables.')
        ], style={'width': '60%', 'color': colors['text'], 'display': 'block', 'font-size': 20,
                  'margin': '0 auto'}),
    ], style={'color': 'Black', 'font-family': 'Arial', 'text-align': 'center', 'border': '2px solid gold',
              'height': '850px'}),
    # scroll here

    # page 2
    html.Div([
        html.Div([
            dcc.Markdown('''
            > To measure the impact of their work, and to optimize how they collect and redistribute their food, 
            > city harvest collects metrics on supply gap, meal gap, and food insecurity population, 
            > for each NTA District in NYC. 
           '''.replace('  ', '')),
        ], style={'color': 'Black', 'font-family': 'Arial', 'margin-top': '200px', 'font-size': 30,
                  'width': '70%', 'display': 'block', 'margin-left': '200px'}),
        html.Div([
            dcc.Markdown('''
               * __Food Insecurity population__ - USDA defines food insecurity as "a  lack of consistent access to enough food for an active, healthy life
               * __Meal Gap__ - Feeding America defines meal gap as " a conversion of the total annual food budget shortfall in a specified area divided by the weighted cost per meal in that area. The meal gap number represents the translation of the food budget shortfall into a number of meals."
               * __Supply Gap__ - This is an internal metric City Harvest uses to quantify the difference of the amount of meals that they need to serve compared to the amount of meals they can serve.
              '''.replace('  ', '')),
        ], style={'color': 'Black', 'font-family': 'Arial', 'font-size': 20,
                  'width': '50%', 'margin': '0 auto', 'display': 'block'}),
    ], style={'color': 'Black', 'font-family': 'Arial', 'width': '100%',
              'margin': '0 auto', 'display': 'block', 'height': '850px', 'border': '2px solid gold'}),

    # Scroll Down

    # Page 3
    html.Div([
        dcc.Markdown('''
           ## Machine Learning Model
           > Using the data City Harvest has on each NTA's __Supply Gap__, __Meal Gap__, and 
           > __Food Insecurity Population__, we combined it with external data we have on 
           > percentage of __household under poverty line__, __unemployment rate__, 
           > percent population with __high school degree__, 
           > percent population with __college degree__, and percent population that are __Not US Citizen__ to make a
           > predictive model that these five features as input and estimates the three food insecurity metrics 

           '''.replace('  ', '')),
    ], style={'color': 'Black', 'font-family': 'Arial',
              'width': '80%', 'margin': '0 auto', 'display': 'block'}),

    html.Div([
        dcc.Markdown('''
       ### For example, 
       > in NTA district __MN33 East Village__, there was __15.5% household under poverty line__, 
       __ 5.7 % Unemployment Rate__, __91.8%__ population with __High School degree__,
       > __67.7% __ population with __College Degree__, and
       __56.9__ population that are __not US Citizen__
        . If we enter those number as the input feature of our model, we could get a predicted 
       > output of *Population*, *Meal Gap*, 'Supply Gap'
       '''.replace('  ', '')),
    ], style={'color': 'Black', 'font-family': 'Arial',
              'width': '80%', 'margin': '0 auto', 'display': 'block'}),

    # Graph Plot of East Village
    html.Div([
        html.Div([
            dcc.Graph(figure=update_graph_1_1(code_default), id='feature-graph-1')
        ], style={'display': 'inline-block', 'width': '40%', 'float': 'left'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=update_graph_1_2(code_default), id='pop-graph-1')
            ], style=dict(width='100%', padding='0px 0px 0px 0px')),
            html.Div([
                dcc.Graph(figure=update_graph_1_3(code_default), id='mealgap-graph-1')
            ], style=dict(width='100%', padding='0px 0px 0px 0px')),
            html.Div([
                dcc.Graph(figure=update_graph_1_4(code_default), id='supplygap-graph-1')
            ], style=dict(width='100%', padding='0px 0px 0px 0px'))
        ], style={'display': 'inline-block', 'width': '60%', 'float': 'right', 'columnCount': 3}),

    ], style={'width': '100%', 'height': '500px', 'display': 'block'}),

    # select existing community
    html.Div([
        dcc.Markdown('''
           __Now, you can select another NTA district and compare it with East Village__
           '''.replace('  ', '')),
    ], style={'color': 'Black', 'font-family': 'Arial', 'font-size': 20, 'text-align': 'center',
              'width': '80%', 'margin': '0 auto', 'display': 'block'}),

    # drop down 1
    html.Div([
        dcc.Dropdown(
            id='nta1',
            options=[{'label': i, 'value': i} for i in available_indicators],
            value='BK72 Williamsburg'
        )], style={'width': '60%', 'display': 'block', 'padding': '50px 0px 50px 0px', 'margin': '0 auto'}),

    # Plot for EV and the selected
    html.Div([
        html.Div([
            dcc.Graph(id='feature-graph-2')
        ], style={'display': 'inline-block', 'width': '40%', 'float': 'left'}),
        html.Div([
            html.Div([
                dcc.Graph(id='pop-graph-2')

            ], style=dict(width='100%', padding='0px 0px 0px 0px')),
            html.Div([
                dcc.Graph(id='mealgap-graph-2')
            ], style=dict(width='100%', padding='0px 0px 0px 0px')),
            html.Div([
                dcc.Graph(id='supplygap-graph-2')
            ], style=dict(width='100%', padding='0px 0px 0px 0px'))
        ], style={'display': 'inline-block', 'width': '60%', 'float': 'right', 'columnCount': 3}),
    ], style={'width': '100%', 'height': '500px', 'display': 'block'}),

    # Create your own
    html.Div([
        dcc.Markdown('''
        Now you can chose  
        percentage of __household under poverty line__, __unemployment rate__, 
        percent population with __high school degree__, 
        percent population with __college degree__, and percent population that are __Not US Citizen__
        to make your own community and see how it compare to the other two real community
        '''.replace('  ', '')),
    ], style={'color': 'Black', 'font-family': 'Arial',
              'width': '80%', 'margin': '0 auto', 'display': 'block', 'font-size': 20}),
    # 5 slider
    html.Div([
        # slider per
        html.Div([
            html.Div([
                html.P('Percentage of household living under poverty line')
            ], style={'fontSize': 14, 'display': 'inline-block', 'width': '20%'}),
            html.Div([dcc.Slider(
                id='per-slider',
                min=df_2['per'].min(),
                max=df_2['per'].max(),
                value=df_2['per'].min(),
                marks={str(per): str(per) for per in df_2['per'].unique()}
            )], style={'display': 'inline-block', 'padding': '10px 20px 20px 20px', 'width': '70%',
                       'float': 'right'})
        ], style={'width': '97%'}),
        # slider un
        html.Div([
            html.Div([
                html.P('Unemployment Rate')
            ], style={'fontSize': 14, 'display': 'inline-block', 'width': '20%'}),
            html.Div([dcc.Slider(
                id='un-slider',
                min=df_2['un'].min(),
                max=df_2['un'].max(),
                value=df_2['un'].min(),
                marks={str(un): str(un) for un in df_2['un'].unique()}
            )], style={'display': 'inline-block', 'padding': '10px 20px 20px 20px', 'width': '70%',
                       'float': 'right'})
        ], style={'width': '97%'}),
        html.Div([
            html.P('   ')
        ], style={'width': '97%'}),  # empty line
        # slider high
        html.Div([
            html.Div([
                html.P('Percent population that has high school degree')
            ], style={'fontSize': 14, 'display': 'inline-block', 'width': '20%'}),
            html.Div([dcc.Slider(
                id='high-slider',
                min=df_2['high'].min(),
                max=df_2['high'].max(),
                value=df_2['high'].min(),
                marks={str(high): str(high) for high in df_2['high'].unique()}
            )], style={'display': 'inline-block', 'padding': '10px 20px 20px 20px', 'width': '70%',
                       'float': 'right'})
        ], style={'width': '97%'}),
        # slider college
        html.Div([
            html.Div([
                html.P('Percent population that has college school degree')
            ], style={'fontSize': 14, 'display': 'inline-block', 'width': '20%'}),
            html.Div([dcc.Slider(
                id='college-slider',
                min=df_2['college'].min(),
                max=df_2['college'].max(),
                value=df_2['college'].min(),
                marks={str(college): str(college) for college in df_2['college'].unique()}
            )], style={'display': 'inline-block', 'padding': '10px 20px 20px 20px', 'width': '70%',
                       'float': 'right'})
        ], style={'width': '97%'}),
        # slider citizen
        html.Div([
            html.Div([
                html.P('Percent population that are not us citizen')
            ], style={'fontSize': 14, 'display': 'inline-block', 'width': '20%'}),
            html.Div([dcc.Slider(
                id='citizen-slider',
                min=df_2['citizen'].min(),
                max=df_2['citizen'].max(),
                value=df_2['citizen'].min(),
                marks={str(citizen): str(citizen) for citizen in df_2['citizen'].unique()}
            )], style={'display': 'inline-block', 'padding': '10px 20px 20px 20px', 'width': '70%',
                       'float': 'right'})
        ], style={'width': '97%'})

    ], style={'width': '100%', 'display': 'block',
              'background-color': colors['background']}),

    # Final Plot
    html.Div([
        html.Div([
            dcc.Graph(id='feature-graph-3')
        ], style={'display': 'inline-block', 'width': '40%', 'float': 'left'}),
        html.Div([
            html.Div([
                dcc.Graph(id='pop-graph-3')

            ], style=dict(width='100%', padding='0px 0px 0px 0px')),
            html.Div([
                dcc.Graph(id='mealgap-graph-3')
            ], style=dict(width='100%', padding='0px 0px 0px 0px')),
            html.Div([
                dcc.Graph(id='supplygap-graph-3')
            ], style=dict(width='100%', padding='0px 0px 0px 0px'))
        ], style={'display': 'inline-block', 'width': '60%', 'float': 'right', 'columnCount': 3}),

    ], style={'width': '100%', 'height': '500px', 'display': 'block'}),

], style={'background-color': colors['background'], 'left': '0'})


# graph 2-1
@app.callback(
    dash.dependencies.Output('feature-graph-2', 'figure'),
    [dash.dependencies.Input('nta1', 'value')])
def update_graph(code2):
    df_nta_1 = df_true[(df_true['code'] == code_default.split()[0])]
    df_nta_2 = df_true[(df_true['code'] == code2.split()[0])]
    code1 = name[(name['NTAcode'] == code_default)]['name']
    return {
        'data': [
            {'x': [df_nta_1.iloc[0, 6],
                   df_nta_1.iloc[0, 5],
                   df_nta_1.iloc[0, 4],
                   df_nta_1.iloc[0, 3],
                   df_nta_1.iloc[0, 2] * 100],
             'y': ['Percentage of none US Citizen', 'Percent with college degree', 'percent with high school degree',
                   'Unemployment Rate', 'Percent house hold living under poverty line'],
             'type': 'bar', 'orientation': 'h', 'name': str(code1.iloc[0]), 'marker': dict(color=colors['b'])},
            {'x': [df_nta_2.iloc[0, 6],
                   df_nta_2.iloc[0, 5],
                   df_nta_2.iloc[0, 4],
                   df_nta_2.iloc[0, 3],
                   df_nta_2.iloc[0, 2] * 100],
             'y': ['Percentage of none US Citizen', 'Percent with college degree', 'percent with high school degree',
                   'Unemployment Rate', 'Percent house hold living under poverty line'],
             'type': 'bar', 'orientation': 'h', 'name': str(code2), 'marker': dict(color=colors['g'])},
        ],
        'layout': go.Layout(
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 1, 'y': 1},
            hovermode='closest',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            xaxis=dict(title='Hover over the bar see attribute !',
                       titlefont=dict(size=20)),
            yaxis=dict(tickfont=dict(size=1))
        )
    }


# graph 2-2
@app.callback(
    dash.dependencies.Output('pop-graph-2', 'figure'),
    [dash.dependencies.Input('nta1', 'value')])
def update_graph(code2):
    df_nta_1 = df_true[(df_true['code'] == code_default.split()[0])]
    df_nta_2 = df_true[(df_true['code'] == code2.split()[0])]
    code1 = name[(name['NTAcode'] == code_default)]['name']
    return {
        'data': [
            {'x': [str(code1.iloc[0])], 'y': [df_nta_1.iloc[0, 7]], 'type': 'bar',
             'name': str(code1.iloc[0]), 'marker': dict(color=colors['b'])},
            {'x': [str(code2)], 'y': [df_nta_2.iloc[0, 7]], 'type': 'bar',
             'name': str(code2), 'marker': dict(color=colors['g'])}],
        'layout': go.Layout(
            legend=dict(x=1, y=1, font=dict(family='sans-serif', size=1)),
            hovermode='closest', title='Food Insecurity Population',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
        )
    }


# graph 2-3
@app.callback(
    dash.dependencies.Output('mealgap-graph-2', 'figure'),
    [dash.dependencies.Input('nta1', 'value')])
def update_graph(code2):
    df_nta_1 = df_true[(df_true['code'] == code_default.split()[0])]
    df_nta_2 = df_true[(df_true['code'] == code2.split()[0])]
    code1 = name[(name['NTAcode'] == code_default)]['name']
    return {
        'data': [
            {'x': [str(code1.iloc[0])], 'y': [df_nta_1.iloc[0, 8]], 'type': 'bar',
             'name': str(code1.iloc[0]), 'marker': dict(color=colors['b'])},
            {'x': [str(code2)], 'y': [df_nta_2.iloc[0, 8]], 'type': 'bar',
             'name': str(code2), 'marker': dict(color=colors['g'])}],
        'layout': go.Layout(
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend=dict(x=1, y=1, font=dict(family='sans-serif', size=1)),
            hovermode='closest', title='Meal Gap',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
        )}


# graph 2-4
@app.callback(
    dash.dependencies.Output('supplygap-graph-2', 'figure'),
    [dash.dependencies.Input('nta1', 'value')])
def update_graph(code2):
    df_nta_1 = df_true[(df_true['code'] == code_default.split()[0])]
    df_nta_2 = df_true[(df_true['code'] == code2.split()[0])]
    code1 = name[(name['NTAcode'] == code_default)]['name']
    return {
        'data': [
            {'x': [str(code1.iloc[0])], 'y': [df_nta_1.iloc[0, 9]], 'type': 'bar',
             'name': str(code1.iloc[0]), 'marker': dict(color=colors['b'])},
            {'x': [str(code2)], 'y': [df_nta_2.iloc[0, 9]], 'type': 'bar',
             'name': str(code2), 'marker': dict(color=colors['g'])}],
        'layout': go.Layout(
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend=dict(x=1, y=1, font=dict(family='sans-serif', size=1)),
            hovermode='closest', title='Supply Gap',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
        )}


# graph 3-1
@app.callback(
    dash.dependencies.Output('feature-graph-3', 'figure'),
    [dash.dependencies.Input('nta1', 'value'),
     dash.dependencies.Input('per-slider', 'value'),
     dash.dependencies.Input('un-slider', 'value'),
     dash.dependencies.Input('high-slider', 'value'),
     dash.dependencies.Input('college-slider', 'value'),
     dash.dependencies.Input('citizen-slider', 'value')])
def update_graph(code2, per, un, high, college, us):
    df_nta_1 = df_true[(df_true['code'] == code_default.split()[0])]
    df_nta_2 = df_true[(df_true['code'] == code2.split()[0])]
    code1 = name[(name['NTAcode'] == code_default)]['name']
    return {
        'data': [
            {'x': [df_nta_1.iloc[0, 6],
                   df_nta_1.iloc[0, 5],
                   df_nta_1.iloc[0, 4],
                   df_nta_1.iloc[0, 3],
                   df_nta_1.iloc[0, 2] * 100],
             'y': ['Percentage of none US Citizen', 'Percent with college degree', 'percent with high school degree',
                   'Unemployment Rate', 'Percent house hold living under poverty line'],
             'type': 'bar', 'orientation': 'h', 'name': str(code1.iloc[0]), 'marker': dict(color=colors['b'])},
            {'x': [df_nta_2.iloc[0, 6],
                   df_nta_2.iloc[0, 5],
                   df_nta_2.iloc[0, 4],
                   df_nta_2.iloc[0, 3],
                   df_nta_2.iloc[0, 2] * 100],
             'y': ['Percentage of none US Citizen', 'Percent with college degree', 'percent with high school degree',
                   'Unemployment Rate', 'Percent house hold living under poverty line'],
             'type': 'bar', 'orientation': 'h', 'name': str(code2), 'marker': dict(color=colors['g'])},
            {'x': [us,
                   college,
                   high,
                   un,
                   per],
             'y': ['Percentage of none US Citizen', 'Percent with college degree', 'percent with high school degree',
                   'Unemployment Rate', 'Percent house hold living under poverty line'],
             'type': 'bar', 'orientation': 'h', 'name': 'Custom', 'marker': dict(color=colors['r'])},
        ],
        'layout': go.Layout(
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 1, 'y': 1},
            hovermode='closest',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            xaxis=dict(title='Hover over the bar see attribute !',
                       titlefont=dict(size=20)),
            yaxis=dict(tickfont=dict(size=1))
        )
    }


# graph 3-2
@app.callback(
    dash.dependencies.Output('pop-graph-3', 'figure'),
    [dash.dependencies.Input('nta1', 'value'),
     dash.dependencies.Input('per-slider', 'value'),
     dash.dependencies.Input('un-slider', 'value'),
     dash.dependencies.Input('high-slider', 'value'),
     dash.dependencies.Input('college-slider', 'value'),
     dash.dependencies.Input('citizen-slider', 'value')
     ])
def update_graph(code2, per, un, high, college, us):
    df_nta_1 = df_true[(df_true['code'] == code_default.split()[0])]
    df_nta_2 = df_true[(df_true['code'] == code2.split()[0])]
    code1 = name[(name['NTAcode'] == code_default)]['name']
    result = updateresult(per, un, high, college, us)
    return {
        'data': [
            {'x': [str(code1.iloc[0])], 'y': [df_nta_1.iloc[0, 7]], 'type': 'bar',
             'name': str(code1.iloc[0]), 'marker': dict(color=colors['b'])},
            {'x': [str(code2)], 'y': [df_nta_2.iloc[0, 7]], 'type': 'bar', 'name': str(code2),
             'marker': dict(color=colors['g'])},
            {'x': ['Custom'], 'y': [result[0, 0]*100], 'type': 'bar', 'name': 'Custom',
             'marker': dict(color=colors['r'])}],
        'layout': go.Layout(
            legend=dict(x=1, y=1, font=dict(family='sans-serif', size=1)),
            hovermode='closest', title='Food Insecurity Population %',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
        )
    }


# graph 3-3
@app.callback(
    dash.dependencies.Output('mealgap-graph-3', 'figure'),
    [dash.dependencies.Input('nta1', 'value'),
     dash.dependencies.Input('per-slider', 'value'),
     dash.dependencies.Input('un-slider', 'value'),
     dash.dependencies.Input('high-slider', 'value'),
     dash.dependencies.Input('college-slider', 'value'),
     dash.dependencies.Input('citizen-slider', 'value')])
def update_graph(code2, per, un, high, college, us):
    df_nta_1 = df_true[(df_true['code'] == code_default.split()[0])]
    df_nta_2 = df_true[(df_true['code'] == code2.split()[0])]
    code1 = name[(name['NTAcode'] == code_default)]['name']
    result = updateresult(per, un, high, college, us)
    return {
        'data': [
            {'x': [str(code1.iloc[0])], 'y': [df_nta_1.iloc[0, 8]], 'type': 'bar', 'name': str(code1.iloc[0]),
             'marker': dict(color=colors['b'])},
            {'x': [str(code2)], 'y': [df_nta_2.iloc[0, 8]], 'type': 'bar', 'name': str(code2),
             'marker': dict(color=colors['g'])},
            {'x': ['Custom'], 'y': [result[0, 1]], 'type': 'bar', 'name': 'Custom',
             'marker': dict(color=colors['r'])}],
        'layout': go.Layout(
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend=dict(x=1, y=1, font=dict(family='sans-serif', size=1)),
            hovermode='closest', title='Meal Gap',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
        )}


# graph 3-4
@app.callback(
    dash.dependencies.Output('supplygap-graph-3', 'figure'),
    [dash.dependencies.Input('nta1', 'value'),
     dash.dependencies.Input('per-slider', 'value'),
     dash.dependencies.Input('un-slider', 'value'),
     dash.dependencies.Input('high-slider', 'value'),
     dash.dependencies.Input('college-slider', 'value'),
     dash.dependencies.Input('citizen-slider', 'value')
     ])
def update_graph(code2, per, un, high, college, us):
    df_nta_1 = df_true[(df_true['code'] == code_default.split()[0])]
    df_nta_2 = df_true[(df_true['code'] == code2.split()[0])]
    code1 = name[(name['NTAcode'] == code_default)]['name']
    result = updateresult(per, un, high, college, us)
    return {
        'data': [
            {'x': [str(code1.iloc[0])], 'y': [df_nta_1.iloc[0, 9]], 'type': 'bar', 'name': str(code1.iloc[0]),
             'marker': dict(color=colors['b'])},
            {'x': [str(code2)], 'y': [df_nta_2.iloc[0, 9]], 'type': 'bar', 'name': str(code2),
             'marker': dict(color=colors['g'])},
            {'x': ['Custom'], 'y': [result[0, 2]], 'type': 'bar', 'name': 'Custom',
             'marker': dict(color=colors['r'])}],
        'layout': go.Layout(
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend=dict(x=1, y=1, font=dict(family='sans-serif', size=1)),
            hovermode='closest', title='Supply Gap',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
        )}


if __name__ == '__main__':
    app.run_server(debug=True)
