import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def run_eda():
    # Load breast cancer data
    data = load_breast_cancer()
    features = pd.DataFrame(data.data, columns=data.feature_names)
    target = pd.Series(data.target, name="target")

    # Combine features and target for EDA
    df = pd.concat([features, target], axis=1)

    st.title("Breast Cancer Data EDA")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["DATAFRAME", "Histograms", "Scatter Plots", "Box Plots", "Correlation Heatmap", "Pair Plot"])

    with tab1:
        st.header("DataFrame")
        st.dataframe(df)

    with tab2:
        st.header("Target value's balance check!!!")

        # Count the occurrences of 0s and 1s
        counts = df['target'].value_counts().reset_index()
        counts.columns = ['target', 'count']

        # Create the bar graph
        balance_fig = px.bar(counts, x='target', y='count', text_auto=True, title='Counts of Not having breast cancer and likely to having breast cancer in target column')
        balance_fig.update_traces(textangle=0, textposition="inside", cliponaxis=False)

        # Show the graph
        st.plotly_chart(balance_fig)

        st.header("Histogram of all features by target")

        cols = st.columns(3)  # Create 3 columns for the histograms
        for i, column in enumerate(features.columns):
            fig = px.histogram(df, x=column, color="target", nbins=50, title=f"Distribution of {column}")
            cols[i % 3].plotly_chart(fig)  # Place the plot in one of the three columns

    with tab3:
        st.header("Scatter Plots")
        cols = st.columns(3)  # Create 3 columns for the scatter plots
        for i, col1 in enumerate(features.columns[:5]):  # Limiting to the first 5 features for brevity
            for col2 in features.columns[:5]:
                if col1 != col2:
                    fig = px.scatter(df, x=col1, y=col2, color="target", title=f"{col1} vs {col2}")
                    cols[i % 3].plotly_chart(fig)  # Place the plot in one of the three columns

    with tab4:
        st.header("Box Plots")
        cols = st.columns(3)  # Create 3 columns for the box plots
        for i, column in enumerate(features.columns):
            fig = px.box(df, x="target", y=column, title=f"Box plot of {column} by Target")
            cols[i % 3].plotly_chart(fig)  # Place the plot in one of the three columns

    with tab5:
        st.header("Correlation Heatmap")

        heatmap_color_scale = st.selectbox("Select Heatmap Colorscale",
                                           ['armyrose', 'agsunset', 'algae', 'amp', 'aggrnyl', 'balance', 'blackbody',
                                            'bluered', 'blues', 'blugrn', 'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu',
                                            'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep', 'delta', 'dense',
                                            'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray',
                                            'greens', 'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno',
                                            'jet',
                                            'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges', 'orrd',
                                            'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg', 'plasma',
                                            'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp',
                                            'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn',
                                            'redor', 'reds', 'solar', 'spectral', 'speed', 'sunset', 'sunsetdark',
                                            'teal',
                                            'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
                                            'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'])

        corr = features.corr()

        # Create the heatmap with annotations
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.index.values,
            y=corr.columns.values,
            colorscale=heatmap_color_scale,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hoverinfo="z"
        ))
        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_nticks=36,
            width=1400,  # Adjust the width as needed
            height=1000  # Adjust the height as needed
        )
        st.plotly_chart(fig)

    # ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn',
    # 'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep', 'delta',
    # 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys', 'haline',
    # 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm',
    # 'oranges', 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg', 'plasma', 'plotly3',
    # 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy',
    # 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal',
    # 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'turbo', 'twilight', 'viridis', 'ylgn',
    # 'ylgnbu', 'ylorbr', 'ylorrd']
    # ##################################################################################################### Plotly
    # provides a variety of colorscales that you can use for your visualizations. Here are some of the available
    # colorscales:
    #
    # Viridis ########
    # Cividis
    # Inferno
    # Magma
    # Plasma
    # Turbo
    # Bluered
    # RdBu #########
    # Picnic ##########
    # Portland #######
    # Jet
    # Hot
    # Gray
    # Electric
    # Rainbow
    # Blackbody
    # Earth  ##########
    # YlGnBu  ######
    # YlOrRd
    ######################################################################################################

    with tab6:
        st.header("Pair Plot")
        pair_plot_color_scale = st.selectbox("Select Pair Plot Colorscale",
                                             ['armyrose', 'agsunset', 'algae', 'amp', 'aggrnyl', 'balance', 'blackbody',
                                              'bluered', 'blues', 'blugrn', 'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu',
                                              'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep', 'delta', 'dense',
                                              'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray',
                                              'greens', 'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno',
                                              'jet', 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
                                              'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
                                              'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd',
                                              'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu',
                                              'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed', 'sunset',
                                              'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal',
                                              'tropic', 'turbid', 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu',
                                              'ylorbr', 'ylorrd'])
        fig = px.scatter_matrix(df, dimensions=features.columns[:5], color="target",
                                title="Scatter Matrix of First 5 Features",
                                width=1600,  # Increase width
                                height=1000,  # Increase height
                                color_continuous_scale=pair_plot_color_scale)
        st.plotly_chart(fig)
