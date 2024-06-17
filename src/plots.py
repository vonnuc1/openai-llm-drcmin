import plotly.express as px

def create_scatterplot(df, x, y):
    ordered_names = list(df.name.unique())
    ordered_names.sort()
    
    fig = px.scatter(df, x=x, y=y, color='name', color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.Pastel + px.colors.qualitative.Safe,
                     hover_data={x: False, y: False, 'name': True, 'date': True, 'title': True}, category_orders={'name': ordered_names})
    
    return fig

def visualize_prompt(fig, df, x, y):
    fig.add_traces(px.scatter(df, x=x, y=y, 
                              hover_data={x: False, y: False, 'name': True, 'date': False, 'title': True}).update_traces(
                                  marker_size=12, marker_color="black", marker_symbol='x').data)
    return fig

def highlight_neighbours(fig, df, x, y):
    fig.add_traces(px.scatter(df, x=x, y=y,
               hover_data={x: False, y: False, 'name': True, 'date': True, 'title': True}).update_traces(
                    marker_color="black", marker_symbol='circle-open').data)
    return fig

def create_scatterplot_with_prompt_and_neighbours(df, x, y, metadatas):
    fig = create_scatterplot(df[:-1], x, y)
    fig = visualize_prompt(fig, df.tail(1), x, y)
    fig = highlight_neighbours(fig, df[df['metadata'].isin(metadatas)], x, y)
    
    return fig