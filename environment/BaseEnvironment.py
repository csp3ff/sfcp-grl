
from environment.PhysicalNetwork import PhysicalNetwork

class BaseEnvironment:
    def __init__(self, config):
        self.config = config
        self.network = PhysicalNetwork(config)

    def compute_result_v0(self, status, sfcr, candidates_sn, candidates_sl):
        """
        Compute the result of the SFCR processing.
        :param sfcr: The SFCR object to be processed.
        :param candidates_sn: A dictionary mapping virtual nodes to selected physical nodes.
        :param candidates_sl: A dictionary mapping virtual links to selected physical links.
        :param status: The status of the SFCR processing (True if successful, False otherwise).
        :return: A tuple containing the status, delay, revenue, cost, revenue/cost ratio, and reward.
        """
        delay, revenue, cost, rc, rewards = 0, [], [], [], []
        for vn in sfcr.vnet.nodes:
            revenue.append((vn.cpu + vn.mem) * sfcr.processing_time)
            sn = self.network.nodes[candidates_sn[vn.id]]
            delay += sn.processing_delay
        sl_max_bw = [0 for _ in range(len(self.network.links))]
        for vl_id, sub_path in candidates_sl.items():
            vl = sfcr.vnet.links[vl_id]
            for sub_pair in sub_path:
                for i in range(1, len(sub_pair)):
                    sl = self.network.links[self.network.g[sub_pair[i-1]][sub_pair[i]]]
                    sl_max_bw[sl.id] = max(sl_max_bw[sl.id], vl.bw)
                    delay += sl.prop_delay
        bw_sum = sum(sl_max_bw)
        cost = [bw_sum * sfcr.processing_time] * len(revenue)
        rc = [r / c if c > 0 else 0 for r, c in zip(revenue, cost)]
        rewards = [0.1 * r for r in rc]
        if not status:
            rewards = [-1 * r for r in rewards]
        result = {
            'status': status,
            'delay': delay,
            'revenue': revenue,
            'cost': cost,
            'revenue_cost_ratio': rc,
            'rewards': rewards
        }
        return result

    def compute_result(self, status, sfcr, candidates_sn, candidates_sl):
        """
        Compute the result of the SFCR processing.
        :param sfcr: The SFCR object to be processed.
        :param candidates_sn: A dictionary mapping virtual nodes to selected physical nodes.
        :param candidates_sl: A dictionary mapping virtual links to selected physical links.
        :param status: The status of the SFCR processing (True if successful, False otherwise).
        :return: A tuple containing the status, delay, revenue, cost, revenue/cost ratio, and reward.
        """
        delay, revenue, cost, rc, rewards = 0, [], [], [], []
        for vn in sfcr.vnet.nodes:
            revenue.append((vn.cpu + vn.mem) * sfcr.processing_time)
            sn = self.network.nodes[candidates_sn[vn.id]]
            delay += sn.processing_delay
        bw_sum = 0
        for vl_id, sub_path in candidates_sl.items():
            vl = sfcr.vnet.links[vl_id]
            for sub_pair in sub_path:
                for i in range(1, len(sub_pair)):
                    sl = self.network.links[self.network.g[sub_pair[i-1]][sub_pair[i]]]
                    bw_sum += vl.bw
                    delay += sl.prop_delay
        cost = [bw_sum * sfcr.processing_time] * len(revenue)
        rc = [r / c if c > 0 else 0 for r, c in zip(revenue, cost)]
        rewards = [0.1 * r for r in rc]
        if not status:
            rewards = [-1 * r for r in rewards]
        result = {
            'status': status,
            'delay': delay,
            'revenue': revenue,
            'cost': cost,
            'revenue_cost_ratio': rc,
            'rewards': rewards
        }
        return result


    def step(self, status, sfcr, sel_sn, sel_sl, timeslot):
        """
        Process a single SFCR in the environment.
        :param sfcr: The SFCR object to be processed.
        :param sel_sn: A dictionary mapping virtual nodes to selected physical nodes.
        :return: dict {'status': bool, 'delay': float, 'revenue': list, 'cost': list, 'rc': list, 'rewards': list}
        """
        if status:
            # node embed
            for vn in sfcr.vnet.nodes:
                sn = self.network.nodes[sel_sn[vn.id]]
                sn_remain_cpu, sn_remain_mem = sn.remain_sn()
                # print(f"Mapping VN {vn.id} to SN {sn.id} with remaining CPU: {vn.cpu}/{sn_remain_cpu}, Memory: {vn.mem}/{sn_remain_mem}")
                if sn_remain_cpu < vn.cpu or sn_remain_mem < vn.mem:
                    raise ValueError("Insufficient resources on the selected physical node.")
                self.map_vn_to_sn(sfcr, vn, sn)
            
            # link mapping
            for vl in sfcr.vnet.links:
                for path in sel_sl[vl.id]:
                    for i in range(1, len(path)):
                        sl = self.network.links[self.network.g[path[i-1]][path[i]]]
                        self.map_vl_to_sl(sfcr, vl, sl)
            for sl in self.network.links:
                assert sl.remain_bw() >= 0, f"Link {sl.id} bandwidth is negative after mapping."
            sfcr.embedded_time = timeslot
            self.network.embedding_sfcr.append(sfcr)
            # raise NotImplementedError("This method should be overridden by subclasses.")
        embedding_result = self.compute_result(status, sfcr, sel_sn, sel_sl)
        assert len(embedding_result['revenue']) == len(sfcr.vnet.nodes), "Revenue length mismatch."
        assert len(embedding_result['cost']) == len(sfcr.vnet.nodes), "Cost length mismatch."
        assert len(embedding_result['revenue_cost_ratio']) == len(sfcr.vnet.nodes), "Revenue_Cost_Ratio length mismatch."
        assert len(embedding_result['rewards']) == len(sfcr.vnet.nodes), "Rewards length mismatch."
        return embedding_result
    
    def reset(self):
        """
        Reset the environment to its initial state.
        This method should be called at the beginning of each episode.
        """
        self.network.clear()
    
    def clear_sfcr(self, timeslot):
        """
        Clear the SFCRs for a specific timeslot.
        :param timeslot: The timeslot for which to clear the SFCRs.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def find_path_shortest(self, src, dst):
        """
        Find the shortest path between two nodes in the network.
        :param src: The source node ID.
        :param dst: The destination node ID.
        :return: A list of node IDs representing the shortest path.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def check_bw_for_path(self, bw, s_path, reserved_bw_by_sl_id=None):
        """
        Check if the bandwidth is sufficient for a given path.
        :param bw: The required bandwidth by v_link.
        :param s_path: The path to check.
        :param reserved_bw_by_sl_id: Optional dict mapping physical link id -> reserved bandwidth
                                     within the current SFC request (SUM semantics).
        :return: True if the bandwidth is sufficient, False otherwise.
        """
        if reserved_bw_by_sl_id is None:
            reserved_bw_by_sl_id = {}
        for i in range(1, len(s_path)):
            sl = self.network.links[self.network.g[s_path[i-1]][s_path[i]]]
            effective_remain = sl.remain_bw() - reserved_bw_by_sl_id.get(sl.id, 0)
            if bw > effective_remain:
                return False
        return True

    def map_vl_to_sl(self, sfcr, vl, sl):
        """
        Map a virtual link to a physical link.
        :param sfcr_id: The ID of the SFCR to which the virtual link belongs.
        :param vl: The virtual link to be mapped.
        :param sl: The physical link to which the virtual link is mapped.
        """
        try:
            # Bandwidth check uses SUM semantics: every mapped virtual link consumes its own bandwidth.
            assert vl.bw <= sl.remain_bw(), "Link Error"
        except AssertionError:
            print("Link Error")
            print(sfcr.id, vl.id, vl.bw, "\n", sl.id, sl.remain_bw())
            raise
        sl.hosting[sfcr].append(vl)
        vl.hosting.append(sl)

    def map_vn_to_sn(self, sfcr, vn, sn):
        """
        Map a virtual link to a physical link.
        :param sfcr_id: The ID of the SFCR to which the virtual link belongs.
        :param vl: The virtual link to be mapped.
        :param sl: The physical link to which the virtual link is mapped.
        """
        c1, c2 = sn.remain_sn()
        assert(c1 >= vn.cpu and c2 >= vn.mem if sfcr not in sn.hosting.keys() else False), "Node Error"
        sn.hosting[sfcr] = vn
        vn.hosting = sn
    
    def unmap_sfcr(self, sfcr):
        """
        Unmap a virtual network from the physical network.
        :param sfcr: The SFCR object containing the virtual network to be unmapped.
        """
        for sn in self.network.nodes:
            if sfcr in sn.hosting:
                del sn.hosting[sfcr]
        for sl in self.network.links:
            if sfcr in sl.hosting:
                del sl.hosting[sfcr]
        assert all(sfcr not in sn.hosting for sn in self.network.nodes), "unmap node fail"
        assert all(sfcr not in sl.hosting for sl in self.network.nodes), "unmap link fail"