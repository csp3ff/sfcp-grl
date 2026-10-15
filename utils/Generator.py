# config
import os

from environment.VirtualNetwork import VituralNetwork
from environment.SFCR import SFCR
import numpy as np

def gen_sfcr_union(start_time):
    reqs = []
    nnn = np.random.randint(50, 80)
    n = np.random.randint(2, 10, size=nnn)
    m = n - 1
    start_t = np.random.randint(low=start_time, high=start_time + 90, size=nnn)
    end_t = start_t + 10
    processing_time = np.random.randint(low=200, high=400, size=nnn)
    for i in range(nnn):
        _sfcr = SFCR()
        _sfcr.id = "sfc_" + str(start_time+i)
        _sfcr.start_t = start_t[i]
        _sfcr.end_t = end_t[i]
        _sfcr.processing_time = processing_time[i]
        vnet = VituralNetwork(n[i],m[i])
        for ni in range(n[i]):
            b = np.random.uniform(60, 200)
            c = np.random.uniform(100, 400)
            vnet.add_node(ni, b, c)
        for v in range(1, n[i]):
            u = np.random.randint(0, v)
            bw = np.random.uniform(50, 300)
            vnet.add_edge(v - 1, u, v, bw)
        _sfcr.vnet = vnet
        _sfcr.src = [vnet.nodes[0]]
        reqs.append(_sfcr)
    reqs.sort(key=lambda x : x.start_t)
    return reqs

def gen_sfcr_random(start, end, arrival_rate):
    reqs = []
    for time_step in range(start, end + 1):
        Ri = np.random.rand()
        if Ri < arrival_rate:
            _sfcr = SFCR()
            _sfcr.id = "sfc_" + str(time_step)
            _sfcr.start_t = time_step
            _sfcr.end_t = time_step + np.random.randint(1, 5)
            _sfcr.processing_time = np.random.randint(200, 400)
            n = np.random.randint(2, 10)
            m = n - 1
            vnet = VituralNetwork(n, m)
            for ni in range(n):
                b = np.random.uniform(60, 200)
                c = np.random.uniform(100, 400)
                vnet.add_node(ni, b, c)
            for v in range(1, n):
                u = np.random.randint(0, v)
                bw = np.random.uniform(50, 300)
                vnet.add_edge(v - 1, u, v, bw)
            _sfcr.vnet = vnet
            _sfcr.src = [vnet.nodes[0]]
            reqs.append(_sfcr)
    return reqs

def save_sfcr(reqs):
    # 将每一个req都写入一个单独的文件，文件名是req.id，文件内容是req的属性
    os.makedirs("reqs", exist_ok=True)
    for req in reqs:
        print(req.id)
        filename = f"reqs/{req.id}.txt"
        with open(filename, "w") as f:
            # 写入基本属性
            f.write(f"{req.id}\n")
            f.write(f"{req.start_t} {req.end_t} {req.processing_time}\n")
            f.write(f"{len(req.vnet.nodes)} {len(req.vnet.links)}\n")
            for node in req.vnet.nodes:
                f.write(f"{node.id} {node.cpu} {node.mem}\n")
            for edge in req.vnet.links:
                f.write(f"{edge.id} {edge.u} {edge.v} {edge.bw}\n")
            for src in req.src:
                f.write(f"{src.id}\n")
            
