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
        #node_sizes.append(5 * (1 + graph.degree[node]))  # Adjust node size based on degree
        node_sizes.append(20)
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
        colorscale= st.session_state['colour_gradient'], #good schemes: Plasma, Haline, 
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
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    fig.update_layout(height = 900)
    return fig

def plot_weighted_graph(graph):
    pos = nx.spring_layout(graph)
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weight = edge[2]['weight']
        edge_text.append(f'Weight: {edge_weight}')  # Include weight in edge text
    
    node_x = []
    node_y = []
    text = []
    node_sizes = []
    degrees = []
    for node in graph.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        text.append(f"User {node[0]}<br>Interests: {', '.join(node[1]['interests'])}")
        node_sizes.append(20)
        degrees.append(graph.degree[node[0]])  # Store degree for use in colorbar
        
    # Edge plots
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.0, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=edge_text  # Assign edge_text to display weights
    )
    
    # Node plots
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=text,
        marker=dict(
            showscale=True,
            colorscale=st.session_state['colour_gradient'],  # Adjust according to your session state
            size=node_sizes,
            color=degrees,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    )
    
    fig.update_layout(height=900)
    return fig


@st.cache_resource()
def make_graph_from_profiles(profiles):
    #st.write(profiles)
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

@st.cache_resource()
def make_weighted_graph_from_profiles(profiles):
    #st.write(profiles)
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
                edge_weight = sum(profiles[i]['interests'][interest] for interest in common_interests)
                graph.add_edge(profiles[i]['id'], profiles[j]['id'], common_interests=common_interests, weight = edge_weight)
    
    return graph



def rearrange_dicts(original_dicts):
    interests_mapping = {}

    for original_dict in original_dicts:
        for interest, value in original_dict["interests"].items():
            if interest not in interests_mapping:
                interests_mapping[interest] = {}
            interests_mapping[interest][original_dict["id"]] = value

    for interest, id_values in interests_mapping.items():
        new_dict = {"id": interest, "interests": id_values}
        yield new_dict

    