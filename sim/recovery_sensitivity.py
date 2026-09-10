"""Matched-fault recovery comparison with a common checkpoint regime.

This extends the existing simulator's experiment, not a second simulator.
Local recovery and masking are deliberately not simulated: the current workload
is one synchronous gang and does not define semantics for retaining worker-local
updates after failure. Inventing those semantics would misstate training progress.
"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from reliability_sim import Config, Sim, latent_events


def experiment():
    runs=[]
    for blast in (1,8,32):
        for reload in (.05,.25,1.):
            for repair in (1.,4.,12.):
                for seed in range(12):
                    cfg=Config(seed=seed,spare_nodes=32,reload_h=reload,repair_h=repair,
                               burst_blast=blast,burst_rate_per_h=0 if blast==1 else .2/blast,
                               ckpt_interval_h=.1,ckpt_write_h=.003,
                               ckpt_overlap_interval_h=.1,ckpt_overlap_write_h=.003)
                    events=latent_events(cfg,np.random.default_rng(seed))
                    results={p:Sim(cfg,p,events).run() for p in ('auto-restart','auto-spares','elastic')}
                    runs.append(dict(config=asdict(cfg),evidence_class='simulated',results=results,
                                     winner=max(results,key=lambda p:results[p]['ettr'])))
    return dict(evidence_class='simulated',input_class='scenario',runs=runs,
                controls=['Identical latent faults and checkpoint interval/write stall across policies.',
                          'Spare cost included in the paid accelerator-hour denominator.',
                          'Burst allocation preserves mean node-failure budget.'],
                uncertainty='Reload, repair and burst size are stress scenarios, not externally fitted distributions.',
                omissions={'localized_recovery':'Synchronous gang model does not establish worker-local state recovery correctness.',
                           'failure_masking':'No model of redundant computation/state or retained training progress; no credible masking benchmark.'})


if __name__=='__main__':
    report=experiment();out=Path('results/recovery-sensitivity.json')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    counts={p:sum(r['winner']==p for r in report['runs']) for p in ('auto-restart','auto-spares','elastic')}
    print(json.dumps(dict(scenarios=len(report['runs']),winners=counts),indent=2))
