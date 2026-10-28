"""Causal candidate planning from public laws or prefixes; no actual future."""
import copy,json,random
from pathlib import Path
from baseline import Policy as Base,PlanningFailure
from core import State
from expected import ExpectedRecovery

PARAMS={'overlap_prior_mix':.15,'smoothing':2.0}

class Kernel(Base,ExpectedRecovery):
    def __init__(self,public,style='expected_marginal',uniform=False,forecast_mode='prefix'):
        super().__init__(public);self.style=style;self.uniform=uniform;self.forecast_mode=forecast_mode
        # Private reference implementation constants; never passed as observations.
        self.decay=.85;self.prior_mix=.15;self.age_power=.25;self.window=8;self.count={i:0. for i in self.support};self.transitions={};self.previous=None;self.setup_expected();self.overlap={}
        for a in self.support:
            values={}
            for b in self.support:
                union=self.anc[a]|self.anc[b];common=self.anc[a]&self.anc[b]
                values[b]=sum(self.nodes[p]['cost'] for p in common)/max(1,sum(self.nodes[p]['cost'] for p in union))
            total=sum(values.values());self.overlap[a]={b:v/max(1e-30,total) for b,v in values.items()}
    def fast_clone(self):
        clone=copy.copy(self);clone.last=list(self.last);clone.count=dict(self.count);clone.transitions=dict(self.transitions);clone.expected_cache={};clone.actions=[]
        # Public graph/ancestry and task-cost memo contain no request arrays and
        # are shared read-only (memoized graph arithmetic has no reward labels).
        return clone
    def predict(self,current):
        if self.uniform:return {t:1/len(self.support) for t in self.support}
        if self.forecast_mode=='law':return self.law_forecast(current)
        total=sum(self.count.values());k=len(self.support)
        freq={t:(1-self.prior_mix)*self.count[t]/max(1e-30,total)+self.prior_mix/k for t in self.support}
        mix=PARAMS['overlap_prior_mix'];prior={t:(1-mix)*freq[t]+mix*self.overlap[current][t] for t in self.support}
        row={t:self.transitions.get((current,t),0.) for t in self.support};total=sum(row.values());smooth=PARAMS['smoothing']
        return {t:(row[t]+smooth*prior[t])/(total+smooth) for t in self.support}
    def observe(self,target):
        for t in self.count:self.count[t]*=self.decay
        for pair in list(self.transitions):
            self.transitions[pair]*=self.decay
            if self.transitions[pair]<1e-8:self.transitions.pop(pair)
        self.count[target]+=1.
        if self.previous is not None:self.transitions[(self.previous,target)]=self.transitions.get((self.previous,target),0.)+1.
        self.previous=target;self.forecast=self.predict(target);self.expected_cache={}
    def device_value(self,i):
        if self.style=='cost_aware':return Base.device_value(self,i)
        if self.style=='frequency_ancestry':
            mass=sum(prob for t,prob in self.forecast.items() if i in self.anc[t]);cost=min(self.ancestor_cost[i],self.nodes[i]['transfer_cost']) if i in self.host else self.ancestor_cost[i]
            return cost*mass/(self.nodes[i]['size']*self.age(i)**self.age_power)
        base=self.expected_recovery(self.device,self.host);without=self.expected_recovery(self.device-{i},self.host)
        return max(0,without-base)/(self.nodes[i]['size']*self.age(i)**self.age_power)
    def host_value(self,i):
        if self.style=='cost_aware':return Base.host_value(self,i)
        with_copy=self.expected_recovery({0},self.host|{i});without=self.expected_recovery({0},self.host-{i})
        return max(0,without-with_copy)/(self.nodes[i]['size']*self.age(i)**self.age_power)
    def want_store(self,i):
        if self.style=='cost_aware':return Base.want_store(self,i)
        missing=self.device-{i};before=self.expected_recovery(missing,self.host);after=self.expected_recovery(missing,self.host|{i})
        return max(0,before-after)*self.window>self.nodes[i]['transfer_cost']


def initialize(public):return Kernel(public,forecast_mode="law")
