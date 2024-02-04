import plotly.graph_objects as go
import networkx as nx
import streamlit as st


def plot_graph(graph):
    pos = nx.spring_layout(graph)
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        #edge_text.append('Hello')
    
    node_x = []
    node_y = []
    text = []
    node_sizes = []
    degrees = []
    for node in graph.nodes():
        #st.write(node)
        x,y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(f"User {node}<br>Interests: {', '.join(graph.nodes[node]['interests'])}")
        node_sizes.append(5 * (1 + graph.degree[node]))  # Adjust node size based on degree
        degrees.append(graph.degree[node])  # Store degree for use in colorbar
        
    
    # Edge plots
    edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1.0, color='#888'),
    hoverinfo='text',
    mode='lines')
    
    # Node plots
    node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text = text,
    #fill = degrees,
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=node_sizes,
        color = degrees,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
            )
        )
    )
    pos = {node: (node_x[i], node_y[i]) for i, node in enumerate(graph.nodes)}
    graph.pos = pos
    node_text = list(text)
    node_trace.text = node_text

    edge_text = list(edge_text)
    edge_trace.text = edge_text
    #st.write(edge_trace.text)

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    

    return fig


@st.cache_resource()
def make_graph_from_profiles(profiles):
    #pos = {'professor-simon-shepherd': (0, 0), 'dr-chris-rushton': (-1, 0.3), 'dr-james-tate': (2, 0.17)}
    graph = nx.Graph()
    

    for profile in profiles:
        graph.add_node(profile['id'], interests=profile['interests'])
    
    #pos = nx.spring_layout(graph)
    #st.write(pos)

    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            set_i = set(profiles[i]['interests'])
            set_j = set(profiles[j]['interests'])
            common_interests = set_i.intersection(set_j)
            if common_interests:
                graph.add_edge(profiles[i]['id'], profiles[j]['id'], common_interests=common_interests)
    
    return graph