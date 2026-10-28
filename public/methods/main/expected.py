"""Causal workload-law forecast and shared-task DAG recovery price surrogate.

These prices rank eviction choices. Actual actions, allocations and charges are
replayed separately; the surrogate ignores possible within-request spill work.
"""
class ExpectedRecovery:
    def setup_expected(self):
        self.bits=[1<<i for i in range(len(self.nodes))];self.n=len(self.nodes)
        self.charge=[nd['cost'] for nd in self.nodes]+[nd.get('transfer_cost',0) for nd in self.nodes];self.mask_cost={0:0};self.expected_cache={};self.forecast={t:1/len(self.support) for t in self.support}
    def task_cost(self,mask):
        if mask not in self.mask_cost:
            todo=mask;value=0
            while todo:
                bit=todo&-todo;value+=self.charge[bit.bit_length()-1];todo^=bit
            if len(self.mask_cost)>10000:self.mask_cost={0:0}
            self.mask_cost[mask]=value
        return self.mask_cost[mask]
    def expected_recovery(self,device,host):
        gm=sum(self.bits[i] for i in device);hm=sum(self.bits[i] for i in host);key=(gm,hm)
        if key not in self.expected_cache:
            plans=[0]*self.n
            for i,nd in enumerate(self.nodes):
                if gm&self.bits[i]:continue
                mask=self.bits[i]
                for p in nd['parents']:mask|=plans[p]
                if hm&self.bits[i] and nd['transfer_cost']<self.task_cost(mask):mask=1<<(self.n+i)
                plans[i]=mask
            self.expected_cache[key]=sum(prob*self.task_cost(plans[t]) for t,prob in self.forecast.items())
        return self.expected_cache[key]
    def law_forecast(self,current,steps=None,regularize=True):
        law=self.public.get('request_distribution',{});kind=law.get('kind');k=len(self.support);window=max(1,int(self.window if steps is None else steps));prob={t:float(t==current) for t in self.support};forecast={t:0. for t in self.support}
        groups=self.public.get('target_groups',[]);membership={t:j for j,group in enumerate(groups) for t in group}
        for horizon in range(1,window+1):
            if kind=='iid_uniform':prob={t:1/k for t in self.support}
            elif kind=='sticky_bursts':
                q=law['sticky_repeat_probability'];prob={t:q*prob[t]+(1-q)/k for t in self.support}
            elif kind=='clustered_shifts' and groups:
                totals=[sum(prob[t] for t in group) for group in groups];phase=law['phase_length']
                if (self.clock+horizon-1)%phase==0:
                    prob={t:(sum(totals)-totals[membership[t]])/((len(groups)-1)*len(groups[membership[t]])) for t in self.support}
                else:
                    q=law['within_cluster_repeat_probability'];prob={t:q*prob[t]+(1-q)*totals[membership[t]]/len(groups[membership[t]]) for t in self.support}
            else:
                total=sum(self.count.values());prob={t:(1-self.prior_mix)*self.count[t]/max(1e-30,total)+self.prior_mix/k for t in self.support}
            for t in self.support:forecast[t]+=prob[t]/window
        # Stationary iid support is known exactly; elsewhere a small prefix
        # frequency component regularizes the short-horizon Markov surrogate.
        if regularize and kind in ('sticky_bursts','clustered_shifts'):
            total=sum(self.count.values())
            forecast={t:(1-self.prior_mix)*forecast[t]+self.prior_mix*self.count[t]/max(1e-30,total) for t in self.support}
        return forecast
