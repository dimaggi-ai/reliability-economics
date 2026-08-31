# Which Recovery Policy Wins in Which Failure Regime? A Reliability-Economics Model for Large GPU Training Clusters

**Margaret Nanyonga**, DIMAGGI AI

> Staged preprint (arXiv-ready). To submit: convert to the arXiv template, add author endorsement, and upload. The claims, figures, and numbers here are reproduced by the accompanying open-source repository (`dimaggi-ai/reliability-economics`, MIT).

## Abstract

At the scale of frontier training, a GPU cluster faults every few hours, so recovery is not an exception but a continuous cost — and the operator's real question is not *whether* to spend on recovery but *on what*: warm spares, elastic degradation, checkpoint cadence, or faster repair. We show that the answer is not a single policy but a **region of a phase diagram** defined by the failure process and the spare-pool size, and that the boundaries are drawable. Using a discrete-event model in which all policies face one shared latent fault timeline (so interruptions are an output, not a policy-independent input), with correlated failure bursts, a finite spare queue whose reserved pool is priced into the envelope, checkpoint cadence as a policy lever with a write cost, and a legal-shape-constrained elastic-shrink cost, we find: (i) holding no reserve always loses to elastic degradation; (ii) elastic-shrink is a strong, robust default; (iii) a right-sized spare pool paired with overlapped checkpointing is the best policy, but only within a band and by a thin margin — and *over-provisioning the pool is charged and hurts*; (iv) the spread between a naive and a well-provisioned posture is ~\$21M/month on one 16,384-GPU cluster. Under a Meta-like configuration the model lands just under Meta's published >90% effective-training-time bound (0.87) while reproducing the ~16–22% of a naive posture, and it reproduces the direction of Meta RSC's inverse-with-scale MTTF; it is honest about which of its inputs a skeptic should push on, and its validation registry publishes the anchors it declines to check.

## 1. Introduction

The public conversation about AI-infrastructure reliability tends to report a headline number — "we achieve X% goodput." That framing hides the decision an operator actually faces. Given a fixed, unavoidable fault rate, the money is in *which recovery mechanism* converts the surviving capacity back into productive work per dollar, and that answer moves with the shape of the failure process. This paper makes the trade-off explicit and quantitative, in the field's own units (ETTR / goodput and dollars), and delivers the result as a phase map rather than a point estimate.

A first-generation version of this model reported a single headline ("a two-node warm-spare pool is the biggest lever, ~83% capacity"). Independent expert review, and a prototype, showed that finding to be an artifact of a single-Poisson, policy-independent fault process and an always-sufficient spare pool. This paper is the corrected model; §6 states the corrections plainly, because pre-empting the critique is part of the contribution.

## 2. Model

One gang-scheduled training job of *N* GPUs runs for a horizon *H*. Hardware faults occur on **one absolute-time latent timeline per random seed**, shared by every policy; a policy differs only in how it detects and recovers, so a policy that recovers well genuinely experiences fewer job interruptions and no policy is scored against another's interruption stream. The failure process is a base rate of single-node faults plus rarer common-cause **bursts** of a blast radius *b* (rack/PDU/switch), with the total node-failure rate held fixed as *b* varies so correlation is isolated from rate.

Five recovery policies are compared: **manual** (page a human), **auto-restart** (no spares), **auto-spares** (a finite warm pool of *k* nodes with three distinct clocks — swap-in-and-reload, repair, requalification (return-to-pool being their sum)), **auto-spares + overlapped checkpointing**, and **elastic-shrink** (run narrower to the next legal data-parallel shape, paying a reshard cost). Checkpoint cadence is a lever with a write cost (blocking vs overlapped/async), giving a Young–Daly optimum. Crucially, the **reserved spare pool is priced into the envelope**: a pool held out of production is a cost, so a larger pool must earn its keep to raise ETTR.

Accounting closes exactly (uptime + downtime = horizon, enforced as a runtime invariant). Metrics: **ETTR** (effective training time ratio = productive ÷ envelope, ≈ Meta's metric) as the headline; MTTF/MTTR/availability as diagnostics; and a dollar layer at a configurable \$/GPU-hour.

## 3. Results — the phase diagram

*(Figure: `figures/phase_diagram.png` — ETTR(spares + overlapped checkpoint) − ETTR(elastic) over blast radius × pool size.)*

The winner is a region, not an ordering. No-spares always loses to elastic. Elastic-shrink is a strong robust default (~0.80 ETTR). A right-sized pool with overlapped checkpointing wins only in a band (roughly *k* matched to the blast radius) and by a thin margin; too small a pool collapses to the repair wait, and too large a pool is dragged down by the idle reservation now charged to the envelope. On a 16,384-GPU fleet at \$2.5/GPU-hour, the naive posture wastes ~\$24.8M/month and the best posture ~\$3.9M/month — a ~\$21M/month spread, of which the automation-of-any-kind step (elastic or right-sized spares) captures ~\$18M and the further tuning ~\$2M.

## 4. Validation

*(Figure: `figures/validation.png`.)* A Meta-like configuration (right-sized pool, overlapped checkpointing, fast detection) reproduces ~0.87 ETTR against Meta's published >90% effective training time on the Llama 3 405B run — and the same model reproduces the ~16–22% ETTR of a naive posture, explaining the apparent paradox (same hardware, different regime). MTTF between job interruptions falls with cluster size, consistent with Meta RSC's ∝1/N scaling (slightly steeper at the top end).

## 5. Related work

Meta's Llama 3 report and RSC reliability study establish the fault rates and the ETTR/MTTF framing; ByteDance's MegaScale and ByteRobust the overlapped/every-step checkpoint regime; CheckFreq, Gemini, and just-in-time checkpointing the cadence–overlap design space; Young and Daly the checkpoint optimum. This work's contribution is the regime-aware *economic* comparison across policies, priced and validated, rather than a single system's headline.

## 6. What a skeptic should attack

Deterministic blast radius and point recovery times (real distributions are heavy-tailed); progress linear in usable width above the legal-shape floor; one training job with no inter-job scheduling and no inference; SDC and stragglers modelled as economic regimes, not devices; and a base rate calibrated to public snapshots. The phase *structure* is robust to these; specific cell values are site-specific and every calibration value is an exposed input.

## 7. Conclusion

Recovery strategy is a design choice against your failure process, not a fixed best practice. Hold a reserve sized to your blast radius or degrade elastically; checkpoint often but overlapped; and do not over-provision the pool. The decision is worth ~\$21M/month on one cluster — and it is drawable, in the operator's own units.
