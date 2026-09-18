import networkx as nx
from typing import List, Dict, Any
from temporal.schema import TemporalRelation, Event

class TemporalSceneGraph:
    """Manages the temporal scene graph using NetworkX."""
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        
    def add_object(self, object_id: str, class_name: str, first_seen: float, last_seen: float):
        """Add or update an object node in the graph."""
        self.graph.add_node(object_id, 
                            class_name=class_name, 
                            first_seen=first_seen, 
                            last_seen=last_seen)
                            
    def add_relation(self, relation: TemporalRelation):
        """Add a temporal relation edge."""
        self.graph.add_edge(
            relation.subject_id, 
            relation.object_id, 
            predicate=relation.predicate,
            start_time=relation.start_time,
            end_time=relation.end_time,
            score=relation.mean_score,
            frames=relation.frame_count
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return nx.node_link_data(self.graph)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalSceneGraph":
        """Deserialize from dictionary."""
        tsg = cls()
        tsg.graph = nx.node_link_graph(data)
        return tsg
        
    def query_time(self, timestamp: float) -> nx.MultiDiGraph:
        """Returns a snapshot of the graph at a specific timestamp."""
        subgraph = nx.MultiDiGraph()
        # Copy nodes
        for node, attr in self.graph.nodes(data=True):
            if attr.get("first_seen", 0) <= timestamp <= attr.get("last_seen", float("inf")):
                subgraph.add_node(node, **attr)
                
        # Copy active edges
        for u, v, k, attr in self.graph.edges(data=True, keys=True):
            if attr["start_time"] <= timestamp <= attr["end_time"]:
                if u in subgraph and v in subgraph:
                    subgraph.add_edge(u, v, key=k, **attr)
                    
        return subgraph
