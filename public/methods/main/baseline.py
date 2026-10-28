"""Causal LRU and cost-aware cache policies with shared-ancestor reconstruction."""
import copy
from core import validate_public,MAX_ACTIONS

class PlanningFailure(RuntimeError):pass

class Policy:
    def __init__(self,public,mode='cost_aware'):
        self.public=copy.deepcopy(validate_public(public));self.nodes=self.public['nodes'];self.mode=mode;self.clock=0;self.last=[0]*len(self.nodes);self.anc=[];self.ancestor_cost=[]
        for i,nd in enumerate(self.nodes):
            a={i}
            for p in nd['parents']:a.update(self.anc[p])
            self.anc.append(a);self.ancestor_cost.append(sum(self.nodes[p]['cost'] for p in a))
        self.support=list(self.public['request_support']);self.total_actions=0
    def observe(self,target):pass
    def recover(self,i,known=None,root_host=False):
        known=self.device-{i} if known is None else known;todo=[i];seen=set();value=0
        while todo:
            p=todo.pop()
            if p in known or p in seen:continue
            seen.add(p);nd=self.nodes[p]
            if p in self.host and (p!=i or root_host) and nd.get('transfer_cost',10**30)<self.ancestor_cost[p]:value+=nd['transfer_cost']
            else:value+=nd['cost'];todo.extend(nd['parents'])
        return value
    def age(self,i):return max(1,self.clock-self.last[i]+1)
    def device_value(self,i):
        if self.mode=='lru':return self.last[i]
        cost=min(self.recover(i),self.nodes[i]['transfer_cost']) if i in self.host else self.recover(i)
        return cost/(self.nodes[i]['size']*self.age(i)**.5)
    def host_value(self,i):return max(0,self.ancestor_cost[i]-self.nodes[i]['transfer_cost'])/(self.nodes[i]['size']*self.age(i)**.5)
    def want_store(self,i):return self.ancestor_cost[i]>2*self.nodes[i]['transfer_cost']
    def emit(self,kind,i):
        if self.total_actions+len(self.actions)>=MAX_ACTIONS:raise PlanningFailure('action limit')
        self.actions.append([kind,i])
        if kind in ('compute','load'):
            self.last[i]=self.clock
            if kind=='compute':
                for p in self.nodes[i]['parents']:self.last[p]=self.clock
    def maybe_store(self,i):
        cap=self.public['host_memory'];nd=self.nodes[i]
        if not cap or i in self.host or nd['size']>cap or not self.want_store(i):return
        selected=[];size=self.host_used;value=self.host_value(i)
        for j in sorted(self.host-self.host_pins,key=lambda j:(self.host_value(j),j)):
            if size+nd['size']<=cap:break
            if self.host_value(j)>value:return
            selected.append(j);size-=self.nodes[j]['size']
        if size+nd['size']>cap:return
        for j in selected:self.host.remove(j);self.host_used-=self.nodes[j]['size'];self.emit('drop_host',j)
        self.host.add(i);self.host_used+=nd['size'];self.emit('store',i)
    def room(self,required,locked):
        while self.used+required>self.public['memory']:
            candidates=self.device-locked-{0}
            if not candidates:raise PlanningFailure('recursive live frontier cannot fit selected device budget')
            victim=min(candidates,key=lambda i:(self.device_value(i),i));self.maybe_store(victim);self.device.remove(victim);self.used-=self.nodes[victim]['size'];self.emit('free',victim)
    def ensure(self,i,locked,reverse=False):
        if i in self.device:self.last[i]=self.clock;return
        nd=self.nodes[i]
        if i in self.host and nd['transfer_cost']<=self.recover(i):
            self.host_pins.add(i)
            try:self.room(nd['size'],locked)
            finally:self.host_pins.remove(i)
            self.device.add(i);self.used+=nd['size'];self.emit('load',i);return
        parents=sorted(nd['parents'],key=lambda p:(-self.recover(p),-self.nodes[p]['size'],p),reverse=reverse);keep=set(locked)
        for p in parents:self.ensure(p,keep,reverse);keep.add(p)
        self.room(nd['size']+nd['scratch'],keep);self.device.add(i);self.used+=nd['size'];self.emit('compute',i)
    def step(self,observation):
        if set(observation)!=set(['target','state']):raise ValueError('only current target and cache state accepted')
        target=observation['target'];snapshot=observation['state']
        if set(snapshot)!=set(['device','host','committed']):raise ValueError('state may not expose future/reward fields')
        if target not in self.support or snapshot['committed']!=self.clock:raise ValueError('invalid target or noncausal prefix counter')
        self.clock+=1;self.observe(target);prior_last=list(self.last)
        for reverse in (False,True):
            self.device=set(snapshot['device']);self.host=set(snapshot['host']);self.used=sum(self.nodes[i]['size'] for i in self.device);self.host_used=sum(self.nodes[i]['size'] for i in self.host);self.host_pins=set();self.actions=[];self.last=list(prior_last)
            try:
                self.ensure(target,{0},reverse);self.last[target]=self.clock;self.emit('commit',target);self.total_actions+=len(self.actions);return self.actions
            except PlanningFailure:
                if reverse:raise
        raise PlanningFailure('no complete step')

def initialize(public):return Policy(public)
def initialize_lru(public):return Policy(public,mode='lru')
