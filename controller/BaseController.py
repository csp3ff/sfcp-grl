from collections import defaultdict
from queue import Queue
import torch
from environment.BaseEnvironment import BaseEnvironment

class BaseController:
    def __init__(self, config):
        self.env = BaseEnvironment(config)

    def clear_sfcr(self, timeslot):
        for sfcr in self.env.network.embedding_sfcr:
            if sfcr.embedded_time + sfcr.processing_time <= timeslot:
                self.env.unmap_sfcr(sfcr)
                self.env.network.embedding_sfcr.remove(sfcr)

    def choose_link(self, sfcr, candidates_sn):
        if len(sfcr.vnet.nodes) <= 1:
            return True, "No Link Need"
        assert candidates_sn is not None, "no candidates, vns is empty?"
        candidates_sl = defaultdict(list)
        reserved_bw_by_sl_id = defaultdict(float)
        que = Queue()
        for vn in sfcr.src:
            que.put(vn)
        fail = 0
        while not que.empty():
            vn = que.get()
            for adj_vn in vn.adj:
                path_vn_adj_vn = self.env.network.findpath(candidates_sn[vn.id], candidates_sn[adj_vn.id])
                vl = sfcr.vnet.links[sfcr.vnet.g[vn.id][adj_vn.id]]
                status = self.env.check_bw_for_path(vl.bw, path_vn_adj_vn, reserved_bw_by_sl_id)
                if status:
                    for i in range(1, len(path_vn_adj_vn)):
                        sl = self.env.network.links[self.env.network.g[path_vn_adj_vn[i-1]][path_vn_adj_vn[i]]]
                        reserved_bw_by_sl_id[sl.id] += vl.bw
                    que.put(adj_vn)
                else:
                    fail = 1
                candidates_sl[sfcr.vnet.g[vn.id][adj_vn.id]].append(path_vn_adj_vn)
        return not fail, candidates_sl

    def gen_mask(self, candidates_sn, vn):
        mask = torch.zeros(len(self.env.network.nodes))
        for sn in self.env.network.nodes:
            if sn.id in candidates_sn.values():
                mask[sn.id] = 1
                continue
            cpu, mem = sn.remain_sn()
            if cpu < vn.cpu or mem < vn.mem:
                mask[sn.id] = 1
        mask = ~mask.bool()
        return mask

    def check_link(self, bw, virtual_sl):
        if len(virtual_sl) == 0:
            return False, []
        fail_sn = []
        for id in range(1, len(virtual_sl)):
            if bw > self.env.network.links[self.env.network.g[virtual_sl[id-1]][virtual_sl[id]]].remain_bw():
                fail_sn.append((virtual_sl[id-1], virtual_sl[id]))
        if len(fail_sn) == 0:
            return True, None
        else:
            return False, fail_sn
