"""Trusted causal request/action replay; no compulsory graph warmup.

Only the parent owns future targets. State snapshots contain current caches and
the committed prefix length, with no cost/reward feedback or request arrays.
"""
import copy

MAX_ACTIONS=2_000_000
class InvalidPlan(ValueError):pass

def integer(x,minimum=0):return type(x) is int and x>=minimum

def validate_public(public):
    allowed={'nodes','memory','host_memory','version','protocol','request_support','target_groups','request_distribution','horizon'}
    if not isinstance(public,dict) or set(public)-allowed:raise ValueError('initialize accepts only the declared public graph/budget/support fields')
    law=public.get('request_distribution',{})
    if set(law)-{'kind','sticky_repeat_probability','phase_length','within_cluster_repeat_probability','cluster_change','target_choice'}:raise ValueError('distribution must not include future requests or private seeds')
    nodes=public['nodes']
    if not isinstance(nodes,list) or len(nodes)<2:raise ValueError('source and operator required')
    for i,node in enumerate(nodes):
        if not integer(node['size'],1) or not integer(node['scratch']) or not integer(node['cost'],0 if i==0 else 1):raise ValueError('invalid analytical node size/scratch/cost')
        ps=node['parents']
        if not isinstance(ps,list) or len(set(ps))!=len(ps) or any(not integer(p) or p>=i for p in ps):raise ValueError('parents must uniquely precede node')
        if i==0 and (ps or node['cost']!=0):raise ValueError('source must be pinned and free to read')
        if i>0 and not ps:raise ValueError('non-source requires a parent')
        if i>0 and not integer(node.get('transfer_cost'),1):raise ValueError('positive one-way transfer cost required')
    if not integer(public['memory'],nodes[0]['size']) or not integer(public['host_memory']):raise ValueError('invalid budgets')
    support=public['request_support']
    if not isinstance(support,list) or not support or len(support)!=len(set(support)) or any(not integer(t) or t>=len(nodes) for t in support):raise ValueError('invalid public target support')
    if law:
        if law.get('kind') not in ('iid_uniform','sticky_bursts','clustered_shifts'):raise ValueError('unknown declared request law')
        for name in ('sticky_repeat_probability','within_cluster_repeat_probability'):
            if name in law and (type(law[name]) not in (int,float) or not 0<=law[name]<=1):raise ValueError('invalid declared probability')
        if law.get('kind')=='sticky_bursts' and 'sticky_repeat_probability' not in law:raise ValueError('public sticky probability required')
        if law.get('kind')=='clustered_shifts':
            if 'within_cluster_repeat_probability' not in law:raise ValueError('public cluster-repeat probability required')
            groups=public.get('target_groups')
            if not isinstance(groups,list) or len(groups)<2 or any(not isinstance(g,list) or not g for g in groups):raise ValueError('public nonempty target clusters required')
            flat=[t for group in groups for t in group]
            if len(flat)!=len(set(flat)) or set(flat)!=set(support):raise ValueError('clusters must partition declared support')
            if not integer(law.get('phase_length'),1):raise ValueError('positive public phase length required')
    if 'horizon' in public and not integer(public['horizon'],1):raise ValueError('positive public episode horizon required')
    if 'protocol' in public and public['protocol']!='causal_activation_cache_v1':raise ValueError('unsupported public protocol')
    return public

class State:
    def __init__(self,public,max_actions=MAX_ACTIONS):
        self.public=copy.deepcopy(validate_public(public));self.nodes=self.public['nodes'];self.device={0};self.host=set();self.used=self.nodes[0]['size'];self.host_used=0;self.committed=0;self.cost=0;self.compute_cost=0;self.transfer_cost=0;self.computes=0;self.loads=0;self.stores=0;self.peak=self.used;self.host_peak=0;self.action_count=0;self.max_actions=max_actions
    def snapshot(self):return dict(device=sorted(self.device),host=sorted(self.host),committed=self.committed)
    @classmethod
    def from_snapshot(cls,public,snapshot):
        """Restore a CURRENT trusted observation for planning simulation only.

        Episode grading still begins at the pinned source. Restore is never an
        action and does not let a submitted plan invent resident tensors.
        """
        state=cls(public)
        if set(snapshot)!={'device','host','committed'}:raise ValueError('only current cache snapshot accepted')
        for name in ('device','host'):
            values=snapshot[name]
            if not isinstance(values,list) or len(values)!=len(set(values)) or any(not integer(i) or i>=len(state.nodes) for i in values):raise ValueError('invalid current cache ids')
        state.device=set(snapshot['device']);state.host=set(snapshot['host'])
        if 0 not in state.device or 0 in state.host or not integer(snapshot['committed']):raise ValueError('pinned source/prefix required')
        state.committed=snapshot['committed'];state.used=sum(state.nodes[i]['size'] for i in state.device);state.host_used=sum(state.nodes[i]['size'] for i in state.host)
        if state.used>public['memory'] or state.host_used>public['host_memory']:raise ValueError('snapshot exceeds budgets')
        state.peak=state.used;state.host_peak=state.host_used;return state
    def serve(self,target,actions):
        if not integer(target) or target not in self.public['request_support']:raise InvalidPlan('request outside declared support')
        if not isinstance(actions,list) or not actions:raise InvalidPlan('step must return actions ending in commit')
        if self.action_count+len(actions)>self.max_actions:raise InvalidPlan('episode action limit')
        committed=False
        for a in actions:
            if committed:raise InvalidPlan('no actions after this request commit')
            if not isinstance(a,list) or len(a)!=2:raise InvalidPlan('action must be [kind,index]')
            kind,i=a
            if not integer(i) or i>=len(self.nodes):raise InvalidPlan('invalid action index')
            nd=self.nodes[i];size=nd['size'];self.action_count+=1
            if kind=='commit':
                if i!=target or i not in self.device:raise InvalidPlan('commit requires current target device resident')
                committed=True;self.committed+=1;continue
            if kind=='free':
                if i==0 or i not in self.device:raise InvalidPlan('cannot free pinned/absent tensor')
                self.device.remove(i);self.used-=size
            elif kind=='compute':
                if i==0 or i in self.device:raise InvalidPlan('compute requires absent non-source tensor')
                if any(p not in self.device for p in nd['parents']):raise InvalidPlan('compute parents absent')
                peak=self.used+size+nd['scratch']
                if peak>self.public['memory']:raise InvalidPlan('device cap includes parents, output and scratch')
                self.peak=max(self.peak,peak);self.device.add(i);self.used+=size;self.cost+=nd['cost'];self.compute_cost+=nd['cost'];self.computes+=1
            elif kind=='store':
                if i==0 or i not in self.device or i in self.host:raise InvalidPlan('store needs device tensor absent from host')
                if self.host_used+size>self.public['host_memory']:raise InvalidPlan('host capacity exceeded')
                self.host.add(i);self.host_used+=size;self.host_peak=max(self.host_peak,self.host_used);self.cost+=nd['transfer_cost'];self.transfer_cost+=nd['transfer_cost'];self.stores+=1
            elif kind=='load':
                if i==0 or i not in self.host or i in self.device:raise InvalidPlan('load needs host tensor absent from device')
                if self.used+size>self.public['memory']:raise InvalidPlan('device capacity exceeded by load')
                self.device.add(i);self.used+=size;self.peak=max(self.peak,self.used);self.cost+=nd['transfer_cost'];self.transfer_cost+=nd['transfer_cost'];self.loads+=1
            elif kind=='drop_host':
                if i==0 or i not in self.host:raise InvalidPlan('cannot drop absent/source host tensor')
                self.host.remove(i);self.host_used-=size
            else:raise InvalidPlan('unknown action')
        if not committed:raise InvalidPlan('missing current request commit')
        return self.snapshot()
    def metrics(self):return dict(cost=self.cost,compute_cost=self.compute_cost,transfer_cost=self.transfer_cost,computes=self.computes,loads=self.loads,stores=self.stores,peak=self.peak,host_peak=self.host_peak,requests=self.committed,action_count=self.action_count)

def run_episode(public,requests,policy):
    """Trusted parent adapter; policy receives fresh detached JSON-like values."""
    state=State(public);trace=[]
    for target in requests:
        observation=dict(target=int(target),state=state.snapshot())
        actions=policy.step(copy.deepcopy(observation));state.serve(target,actions);trace.append(actions)
    return state.metrics(),trace

def replay(public,requests,trace):
    if not isinstance(trace,list) or len(trace)!=len(requests):raise InvalidPlan('one action list per request required')
    state=State(public)
    for target,actions in zip(requests,trace):state.serve(target,actions)
    return state.metrics()
