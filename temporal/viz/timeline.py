import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List
from temporal.schema import TemporalRelation

def render_timeline(relations: List[TemporalRelation], video_duration: float) -> go.Figure:
    """Renders a Plotly Gantt chart of relation intervals."""
    if not relations:
        return go.Figure()
        
    data = []
    for r in relations:
        # We need datetime objects for plotly timeline, or we can just use bar chart for seconds
        data.append(dict(
            Pair=f"{r.subject_id} - {r.object_id}",
            Predicate=r.predicate,
            Start=r.start_time,
            End=r.end_time,
            Duration=r.end_time - r.start_time
        ))
        
    df = pd.DataFrame(data)
    
    # We use a horizontal bar chart
    fig = px.timeline(df, x_start="Start", x_end="End", y="Pair", color="Predicate")
    # For numeric x-axis in px.timeline, we must convert start/end to datetime if we use px.timeline natively
    # BUT plotly timeline expects datetimes. Let's do a simple bar chart.
    
    # Simpler approach without datetime:
    fig = go.Figure()
    for pred, group in df.groupby("Predicate"):
        fig.add_trace(go.Bar(
            y=list(group["Pair"]),
            x=list(group["Duration"]),
            base=list(group["Start"]),
            name=pred,
            orientation='h',
            text=list(group["Predicate"]),
            textposition='inside'
        ))
        
    fig.update_layout(
        barmode='overlay',
        xaxis_title="Time (seconds)",
        yaxis_title="Object Pairs",
        xaxis=dict(range=[0, video_duration]),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )
    
    return fig
