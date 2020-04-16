import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode()


def style_dataframe(df: pd.DataFrame, style):
    df = df.style.set_table_styles(style) \
                 .background_gradient(cmap='Blues')
    return df


def barplot(df: pd.DataFrame, col: str, title: str, xlabel: str, ylabel: str):
    bar_values = df[col].value_counts().sort_values(ascending=True)
    x = bar_values.values.tolist()
    y = [i.capitalize() for i in bar_values.index.tolist()]
    colors = ['lightblue'] * len(x)
    colors[-1] = 'mediumblue'
    trace = go.Bar(x=x,
                   y=y,
                   marker_color=colors,
                   orientation='h')
    data = go.Data([trace])
    layout = go.Layout(title=title,
                       xaxis={'title': xlabel},
                       yaxis={'title': ylabel},
                       width=700,
                       height=500)
    figure = go.Figure(data=data, layout=layout)
    return py.iplot(figure)
