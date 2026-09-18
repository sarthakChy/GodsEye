import plotly.graph_objects as go
import networkx as nx

def render_graph(G: nx.MultiDiGraph) -> go.Figure:
    """Renders a NetworkX scene graph into a Plotly figure."""
    if len(G.nodes) == 0:
        return go.Figure()
        
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    edge_text = []
    
    # We want to put labels on the edges, so we calculate midpoints
    mid_x = []
    mid_y = []
    
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        edge_text.append(data.get("predicate", ""))
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    edge_label_trace = go.Scatter(
        x=mid_x, y=mid_y,
        mode='text',
        text=edge_text,
        textposition='top center',
        hoverinfo='none',
        textfont=dict(color='green', size=10)
    )

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='bottom center',
        marker=dict(
            showscale=False,
            color='lightblue',
            size=20,
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig
