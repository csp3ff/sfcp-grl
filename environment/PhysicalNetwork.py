
from collections import defaultdict
import os
import pickle
import networkx as nx
from networkx.exception import NetworkXNoPath
import numpy as np
import torch

class sNode:
    def __init__(self, id, cpu, mem, delay, px, py, pz, do):
        self.id = id
        self.cpu = cpu
        self.mem = mem
        self.processing_delay = delay
        self.px = px
        self.py = py
        self.pz = pz
        self.do = do

class PhysicalNode(sNode):
    def __init__(self, id, cpu, mem, delay, px, py, pz, do):
        super().__init__(id, cpu, mem, delay, px, py, pz, do)
        self.adj = []
        self.hosting = dict()
    
    def remain_sn(self):
        cpu, mem = self.cpu, self.mem
        for sfcr, vn in self.hosting.items():
            cpu -= vn.cpu
            mem -= vn.mem
        assert cpu >= 0, f"Node {self.id} CPU is negative: {cpu}"
        assert mem >= 0, f"Node {self.id} Memory is negative: {mem}"
        return cpu, mem

class sLink:
    def __init__(self, link_id, src_node, dst_node, bandwidth, delay):
        self.id = link_id
        self.u = src_node
        self.v = dst_node
        self.bw = bandwidth
        self.prop_delay = delay

class PhysicalLink(sLink):
    def __init__(self, link_id, src_node, dst_node, bandwidth, delay):
        super().__init__(link_id, src_node, dst_node, bandwidth, delay)
        self.hosting = defaultdict(list)

    def remain_bw(self):
        bw = self.bw
        for sfcr, vls in self.hosting.items():
            # bw -= max([vl.bw for vl in vls])
            bw -= sum([vl.bw for vl in vls])
        assert bw >= 0, f"Link {self.id} bandwidth is negative: {bw}"
        return bw

class PhysicalNetwork:
    def __init__(self, config):
        self.feature_dim = 5
        self.path = config['net_path']
        self.nodes = None
        self.links = None
        self.g = None
        self.dist = None
        self.Graph = self.build_network()
        self.embedding_sfcr = []
        # print(f"Node number: {len(self.nodes)}, Link number: {len(self.links)}")
        # for nodes in self.nodes:
        #     print(nodes.id, nodes.cpu, nodes.mem, nodes.processing_delay, nodes.px, nodes.py, nodes.pz, nodes.do)
        # for links in self.links:
        #     print(links.id, links.u, links.v, links.bw, links.prop_delay)
    
    def clear(self):
        self.embedding_sfcr = []
        for sn in self.nodes:
            sn.hosting = dict()
        for sl in self.links:
            sl.hosting = defaultdict(list)

    def build_network(self):
        Graph = nx.Graph()
        with open(self.path, 'rb') as f:
            Graph = pickle.load(f)
        self.nodes = [None for _ in range(len(Graph.nodes))]
        for node in Graph.nodes(data=True):
            self.nodes[node[0]] = PhysicalNode(node[0], node[1]['cpu'], node[1]['memory'], node[1]['trans_delay'], node[1]['px'], node[1]['py'], node[1]['pz'], node[1]['domain'])
        self.g = [[None for _ in range(len(self.nodes))] for _ in range(len(self.nodes))]
        self.links = [None for _ in range(len(Graph.edges))]
        for index, edge in enumerate(Graph.edges(data=True)):
            self.g[edge[0]][edge[1]] = index
            self.g[edge[1]][edge[0]] = index
            self.links[index] = PhysicalLink(index, edge[0], edge[1], edge[2]['bandwidth'], edge[2]['prop_delay'])
            self.nodes[edge[0]].adj.append(self.nodes[edge[1]])
            self.nodes[edge[1]].adj.append(self.nodes[edge[0]])
        self.dist = [[None for _ in range(len(self.nodes))] for _ in range(len(self.nodes))]
        return Graph
    
    def findpath(self, u, v):
        try:
            path = nx.shortest_path(self.Graph, source=u, target=v)
        except NetworkXNoPath:
            path = []
        return path
    
    def get_feature_matrix(self, sfcr, candidates_sn, vn):
        feature_matrix = []
        temp = dict()
        for _vn, sn in candidates_sn.items():
            temp[sn] = _vn
        for sn in self.nodes:
            cpu, mem = sn.remain_sn()
            if sn in temp.keys():
                cpu -= sfcr.vnet.nodes[temp[sn]].cpu
                mem -= sfcr.vnet.nodes[temp[sn]].mem
            cpu = (cpu - vn.cpu)
            mem = (mem - vn.mem)
            sl_sum = 0.0
            for sn_adj in sn.adj:
                link_id = self.g[sn.id][sn_adj.id]
                sl_sum += self.links[link_id].remain_bw()
            if len(sn.adj):
                sl_sum = sl_sum / len(sn.adj)
            degree = len(sn.adj)
            distance = 0.0
            if vn.pre_vn:
                pre_sn_id = candidates_sn[vn.pre_vn.id]
                if pre_sn_id != sn.id:
                    if self.dist[pre_sn_id][sn.id] == None:
                        self.dist[pre_sn_id][sn.id] = nx.shortest_path_length(self.Graph, pre_sn_id, sn.id)
                        self.dist[sn.id][pre_sn_id] = self.dist[pre_sn_id][sn.id]
                    distance = self.dist[pre_sn_id][sn.id]
            feature_matrix.append([cpu, mem, sl_sum, degree, distance])
        return feature_matrix

    def get_adjacency_matrix(self,network=None):
        N = len(self.nodes)
        adj = np.zeros((N, N), dtype=np.float32)
        for link in self.links:
            u = link.u
            v = link.v
            adj[u][v] = adj[v][u] = 1
        return adj
    
    def get_edge_index(self):
        edge_index = []
        for link in self.links:
            edge_index.append([link.u, link.v])
            edge_index.append([link.v, link.u])
        return edge_index