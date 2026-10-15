
import os
import pickle
import networkx as nx

class vNode:
    def __init__(self, id, cpu, mem):
        self.id = id
        self.cpu = cpu
        self.mem = mem
        self.do = None

class VirtualNode(vNode):
    def __init__(self, id, cpu, mem):
        super().__init__(id, cpu, mem)
        self.adj = []
        self.hosting = None
        self.pre_vn = None

class Link:
    def __init__(self, link_id, src_node, dst_node, bandwidth):
        self.id = link_id
        self.u = src_node
        self.v = dst_node
        self.bw = bandwidth

class VirtualLink(Link):
    def __init__(self, link_id, src_node, dst_node, bandwidth):
        super().__init__(link_id, src_node, dst_node, bandwidth)
        self.hosting = []

class VituralNetwork:
    def __init__(self, size_n, size_l):
        self.g = [[None for _ in range(size_n)] for _ in range(size_n)]
        self.nodes = [None for _ in range(size_n)]
        self.links = [None for _ in range(size_l)]

    def add_node(self, node_id, cpu, mem):
        self.nodes[node_id] = VirtualNode(node_id, cpu, mem)

    def add_edge(self, edge_id, node_id_u, node_id_v, bandwidth):
        self.links[edge_id] = VirtualLink(edge_id, node_id_u, node_id_v, bandwidth)
        self.g[node_id_u][node_id_v] = edge_id
        self.nodes[node_id_u].adj.append(self.nodes[node_id_v])
        self.nodes[node_id_v].pre_vn = self.nodes[node_id_u]