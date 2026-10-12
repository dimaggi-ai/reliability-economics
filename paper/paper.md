# Which Recovery Policy Wins in Which Failure Regime? A Reliability-Economics Model for Large GPU Training Clusters

**Margaret Nanyonga**, DIMAGGI AI

> Staged preprint (arXiv-ready). To submit: convert to the arXiv template, add author endorsement, and upload. The claims, figures, and numbers here are reproduced by the accompanying open-source repository (`dimaggi-ai/reliability-economics`, MIT).

## Abstract

At the scale of frontier training, a GPU cluster faults every few hours, so recovery is not an exception but a continuous cost — and the operator's real question is not *whether* to spend on recovery but *on what*: warm spares, elastic degradation, checkpoint cadence, or faster repair. We show that the answer is not a single policy but a **region of a phase diagram** defined by the failure process and the spare-pool size, and that the boundaries are drawable. Using a discrete-event model in which all policies face one shared latent fault timeline (so interruptions are an output, not a policy-independent input), with correlated failure bursts, a finite spare queue whose reserved pool is priced into the envelope, checkpoint cadence as a policy lever with a write cost, and a legal-shape-constrained elastic-shrink cost, we find: (i) holding no reserve loses to elastic degradation in the modeled sweep; (ii) elastic-shrink is competitive across the modeled sweep; (iii) a right-sized spare pool paired with overlapped checkpointing is the best policy, but only within a band and by a thin margin — and *over-provisioning the pool is charged and hurts*; (iv) the spread between a naive and a well-provisioned posture is ~\$21M/month on one 16,384-GPU cluster. A provisioned configuration yields a local reserved-envelope productive ratio of ~0.87 versus ~0.16–0.22 for naive recovery; these are not interchangeable with Meta's published training-time measures. The model reproduces the direction of Meta RSC's inverse-with-scale MTTF; it is honest about which of its inputs a skeptic should push on, and its validation registry publishes the anchors it declines to check.

## 1. Introduction

The public conversation about AI-infrastructure reliability tends to report a headline number — "we achieve X% goodput." That framing hides the decision an operator actually faces. Given a fixed, unavoidable fault rate, the money is in *which recovery mechanism* converts the surviving capacity back into productive work per dollar, and that answer moves with the shape of the failure process. This paper makes the trade-off explicit and quantitative, using a reserved-envelope productive ratio and GPU-hour valuation, and delivers the result as a phase map rather than a point estimate.

A first-generation version of this model reported a single headline ("a two-node warm-spare pool is the biggest lever, ~83% capacity"). Independent expert review, and a prototype, showed that finding to be an artifact of a single-Poisson, policy-independent fault process and an always-sufficient spare pool. This paper is the corrected model; §6 states the corrections plainly, because pre-empting the critique is part of the contribution.

## 2. Model

One gang-scheduled training job of *N* GPUs runs for a horizon *H*. Hardware faults occur on **one absolute-time latent timeline per random seed**, shared by every policy; a policy differs only in how it detects and recovers, so interruptions and their duration depend on the policy. Faster recovery can expose more separate, shorter interruptions while lengthy outages coalesce faults; fewer counted interruptions alone is not a reliability advantage. No policy is scored against another's interruption stream. The failure process is a base rate of single-node faults plus rarer common-cause **bursts** of a blast radius *b* (rack/PDU/switch), with the total node-failure rate held fixed as *b* varies so correlation is isolated from rate.

Five recovery policies are compared: **manual** (page a human), **auto-restart** (no spares), **auto-spares** (a finite warm pool of *k* nodes with three distinct clocks — swap-in-and-reload, repair, requalification (return-to-pool being their sum)), **auto-spares + overlapped checkpointing**, and **elastic-shrink** (run narrower to the next legal data-parallel shape, paying a reshard cost). Checkpoint cadence is a lever with a write cost (blocking vs overlapped/async), giving a Young–Daly optimum. Crucially, the **reserved spare pool is priced into the envelope**: a pool held out of production is a cost, so a larger pool must earn its keep to raise ETTR.

Accounting closes exactly (uptime + downtime = horizon, enforced as a runtime invariant). The headline ratio is productive GPU-hours divided by the active-plus-spare reserved envelope (legacy `ettr` field), with MTTF/MTTR/availability as diagnostics and a configurable GPU-hour valuation. Meta's ETTR includes eligible queue waiting ([reference 2](../REFERENCES.md), §II-D), whereas this model has no scheduling queue and charges spares in its denominator; the measures are not interchangeable.

## 3. Results — the phase diagram

*(Figure: `figures/phase_diagram.png` — ETTR(spares + overlapped checkpoint) − ETTR(elastic) over blast radius × pool size.)*

The winner is a region, not an ordering. No-spares loses to elastic in the modeled sweep. Elastic-shrink is competitive across the modeled sweep (~0.80 in the local ratio). A right-sized pool with overlapped checkpointing wins only in a band (roughly *k* matched to the blast radius) and by a thin margin; too small a pool collapses to the repair wait, and too large a pool is dragged down by the idle reservation now charged to the envelope.

<!-- economics-headline:start -->
**Committed independent-failure scenario:** 16,384 active GPUs, 30 days (720 hours), seeds 0–7, valued at an assumed $2.50/GPU-hour.

Mean modeled nonproductive-capacity value: manual/no spares **$24,816,774.875**; elastic/no spares **$5,907,455.875**; automated recovery with eight spare nodes and overlapped checkpointing **$3,862,799.875**.

Subtracting the committed `waste_dollars` values gives **$20,953,975.000** versus manual (about $21M), and **$2,044,656.000** versus elastic (**$2,044,656** rounded to whole dollars). These are modeled capacity valuations, not realized cash savings.
<!-- economics-headline:end -->

The eight spare nodes are an additional 64 GPUs, charged $115,200 over 720 hours within the nonproductive-capacity value. Manual uses blocking checkpoints; both elastic and the best spare policy use overlapped checkpoints, so the manual contrast combines several policy changes. Failure rate, timing, checkpoint, rounding and cost exclusions are specified in the [study's economics section](../docs/study.md#the-economics). The corrected elastic gap is approximately $2.04M; the earlier $2,012,974 rounded gap belongs to the prior public snapshot and its repair-accounting defect, documented there. The manual gap remains $20,953,975.

## 4. Validation

*(Figure: `figures/validation.png`.)* A provisioned configuration yields a modeled reserved-envelope productive ratio of ~0.87, while naive recovery yields ~0.16–0.22. Meta's published >90% effective training time supplies external context; differing denominators and workloads prevent treating that comparison as quantitative validation. The elastic MTTF diagnostic falls with cluster size under the imposed fault rate and serialized reshard stops; this is an internal model relationship, not a fitted reproduction of Meta RSC's scaling curve. See the study for the denominator and mechanism.

## 5. Related work

Meta's Llama 3 report and RSC reliability study establish the fault rates and the ETTR/MTTF framing; ByteDance's MegaScale and ByteRobust the overlapped/every-step checkpoint regime; CheckFreq, Gemini, and just-in-time checkpointing the cadence–overlap design space; Young and Daly the checkpoint optimum. This work's contribution is the regime-aware *economic* comparison across policies, priced and validated, rather than a single system's headline.

## 6. What a skeptic should attack

Deterministic blast radius and point recovery times (real distributions are heavy-tailed); progress linear in usable width above the legal-shape floor; one training job with no inter-job scheduling and no inference; SDC and stragglers modelled as economic regimes, not devices; and a base rate calibrated to public snapshots. Phase structure and specific cell values remain conditional on these assumptions; every calibration value is an exposed input.

## 7. Conclusion

Recovery strategy is a design choice against your failure process, not a fixed best practice. Hold a reserve sized to your blast radius or degrade elastically; checkpoint often but overlapped; and do not over-provision the pool. In the stated independent-failure scenario, the modeled nonproductive-capacity difference is ~\$21M over 30 days — and it is drawable, in the operator's own units.
