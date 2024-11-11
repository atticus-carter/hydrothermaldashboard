import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# External stylesheet for Ubuntu font
external_stylesheets = [
    {
        'href': 'https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&display=swap',
        'rel': 'stylesheet'
    },
    dbc.themes.BOOTSTRAP
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Hydrothermal Vent Analysis'

# Custom CSS to use Ubuntu font
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        body {
            font-family: 'Ubuntu', sans-serif;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

# Placeholder for uploaded CSV data
data = {}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Hydrothermal Vent Analysis Dashboard"), width=12, className='text-center mt-4 mb-2')
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select CSV Files')
                ]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                    'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '20px'
                },
                multiple=True
            ),
            html.Div(id='output-data-upload')
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='faunal-distribution'), width=6),
        dbc.Col(dcc.Graph(id='variable-importance'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='cca-plot'), width=6),
        dbc.Col(dcc.Graph(id='diversity-metrics'), width=6)
    ]),
    dbc.Row([
        dbc.Col(html.Pre(id='classification-report', style={'whiteSpace': 'pre-wrap'}), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='vent-specific-distribution'), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='segment-specific-distribution'), width=12)
    ])
])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
    except Exception as e:
        print(e)
        return None


@app.callback(
    [
        Output('faunal-distribution', 'figure'),
        Output('variable-importance', 'figure'),
        Output('cca-plot', 'figure'),
        Output('diversity-metrics', 'figure'),
        Output('classification-report', 'children'),
        Output('vent-specific-distribution', 'figure'),
        Output('segment-specific-distribution', 'figure')
    ],
    [Input('upload-data', 'contents')],
    [Input('upload-data', 'filename')]
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        dataframes = []
        for contents, filename in zip(list_of_contents, list_of_names):
            df = parse_contents(contents, filename)
            if df is not None:
                dataframes.append(df)
        
        # Assume we're working with a concatenated dataset from all CSVs
        full_df = pd.concat(dataframes, axis=0)

        # Reshape the data to extract relevant information for each segment
        segments = ['Segment1', 'Segment2', 'Segment3', 'Segment4']
        vent_data = []
        for index, row in full_df.iterrows():
            vent_name = row['VentName']
            for segment in segments:
                segment_data = {
                    'VentName': vent_name,
                    'Zone': segment,
                    'Taxonomic_Group': row[f'{segment}_Taxonomic_Group'],
                    'Faunal_Density': row[f'{segment}_Faunal_Density'],
                    'Coverage': row[f'{segment}_Coverage'],
                    'Clustering': row[f'{segment}_Clustering'],
                    'Substrate_Type': row[f'{segment}_Substrate_Type'],
                    'Proximity_to_Flow': row[f'{segment}_Proximity_to_Flow'],
                    'Elevation': row[f'{segment}_Elevation'],
                    'Slope': row[f'{segment}_Slope'],
                    'Formation_Type': row[f'{segment}_Formation_Type'],
                    'Surface_Texture': row[f'{segment}_Surface_Texture']
                }
                vent_data.append(segment_data)

        segment_df = pd.DataFrame(vent_data)

        # Drop rows with NaN values to avoid errors during training
        segment_df = segment_df.dropna()

        # Faunal Distribution Plot
        fig_faunal = px.bar(segment_df, x='Zone', y='Faunal_Density', color='Taxonomic_Group',
                            title="Faunal Distribution Across Zones")

        # Random Forest Classifier Feature Importance
        X = segment_df.drop(columns=['Taxonomic_Group'])
        y = segment_df['Taxonomic_Group']
        X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        importance = rf.feature_importances_
        feature_names = X.columns
        fig_importance = px.bar(x=feature_names, y=importance, title="Variable Importance from Random Forest Classifier",
                                labels={'x': 'Feature', 'y': 'Importance'})

        # Canonical Correspondence Analysis (CCA)
        # For demonstration, we're using a PCA equivalent to approximate CCA visually
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(segment_df.select_dtypes(include=np.number))
        pca = sm.PCA(scaled_data, ncomp=2)
        pca_results = pca.factors
        fig_cca = px.scatter(x=pca_results[:, 0], y=pca_results[:, 1], title="Canonical Correspondence Analysis (Approximation)",
                             labels={'x': 'CCA Axis 1', 'y': 'CCA Axis 2'})

        # Diversity Metrics
        diversity_df = segment_df.groupby('Zone')['Taxonomic_Group'].nunique().reset_index()
        fig_diversity = px.bar(diversity_df, x='Zone', y='Taxonomic_Group',
                               title="Diversity Metrics by Zone", labels={'Taxonomic_Group': 'Unique Species Count'})

        # Random Forest Classification Report
        y_pred = rf.predict(X_test)
        classification_rep = classification_report(y_test, y_pred)

        # Vent-Specific Faunal Distribution
        fig_vent_specific = px.sunburst(segment_df, path=['VentName', 'Zone', 'Taxonomic_Group'], values='Faunal_Density',
                                        title="Faunal Type Distribution by Vent and Zone")

        # Segment-Specific Distribution
        fig_segment_specific = px.bar(segment_df, x='Zone', y='Faunal_Density', color='VentName',
                                      facet_col='Taxonomic_Group', title="Faunal Density by Zone for Each Vent")

        return fig_faunal, fig_importance, fig_cca, fig_diversity, classification_rep, fig_vent_specific, fig_segment_specific

    # Return empty outputs if no data uploaded
    return go.Figure(), go.Figure(), go.Figure(), go.Figure(), "", go.Figure(), go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)

# pip install requirements
# dash, dash-bootstrap-components, pandas, plotly, statsmodels, scikit-learn, matplotlib, numpy
