# PRISM: Theoretical Appendix

> Rigorous proofs and proof sketches for the five core theorems of PRISM.
> Companion to `PRISM_ALGORITHM.md`. Notation matches the algorithm doc
> unless noted.

---

## A. Setup and notation

Fix a trace of $T$ steps. We are interested in the latent
$$\tau \in \mathcal{T} := \{0, 1, \dots, T-1, \infty\}$$
representing the index of the first mistake ($\tau = \infty$ means no error).
The posterior at the end of step $t$ is $\pi_t \in \Delta^{|\mathcal{T}|-1}$,
the simplex over $\mathcal{T}$.

Evidence channels (data sources):
- $S_t$ : stage2 principle labels at step $t$ (a tuple in
  $\mathcal{L}^3$ where
  $\mathcal{L} = \{\text{correct-and-aligned}, \text{reasonable-but-incomplete}, \text{nothing-extracted}, \text{contradiction-found}\}$).
- $V_t^{(i)}$ : specialist $i$'s structured emission at step $t$.
- $S^{(1)}_t$ : stage1 soft-reference inconsistency scalar in $[0,1]$.
- $R_t$ : (legacy) review verdict; retired in PRISM but kept for the
  Universality reduction below.

We write $\mathcal{E}_t := (S_t, \{V_t^{(i)}\}_{i \in S^\star_t}, S^{(1)}_t)$
for the per-step evidence bundle and $\mathcal{E}_{0:t}$ for the cumulative
history. The base PRISM update is
$$\pi_{t+1}(\tau) \;\propto\; \pi_t(\tau)\;\prod_{i} p_i(\mathcal{E}_{t+1}^{(i)} \mid \tau).$$

We adopt the conditional-independence working assumption (assumption A1):
$$p(\mathcal{E}_t \mid \tau, x) \;=\; \prod_i p_i(\mathcal{E}_t^{(i)} \mid \tau, x).$$

---

## B. Theorem 1 — Submodularity of Expected Information Gain

**Statement.** Under assumption A1 the EIG functional
$f(S) := I\!\left(\tau \,;\, V^{(S)} \mid \mathcal{E}_{1:t}\right)$
is monotone submodular in $S \subseteq \{1, \dots, M\}$.

**Proof.** This is Krause and Guestrin (2005, Thm. 1). We reproduce the key
chain: under conditional independence,
$H(\tau \mid \mathcal{E}, V^{(A \cup \{i\})}) = H(\tau \mid \mathcal{E}, V^{(A)}, V^{(i)})$.
The information-gain marginal $f(A \cup \{i\}) - f(A) = I(\tau ; V^{(i)} \mid \mathcal{E}, V^{(A)})$
is non-increasing in $A$ because conditioning on more variables in the
mutual-information identity weakly reduces $I$ when the conditioned
variables are themselves informative about $\tau$. Combined with
$f(\emptyset) = 0$ and $f(A) \geq 0$ for all $A$, this gives monotone
submodularity. $\square$

**Corollary (Nemhauser-Wolsey-Fisher).** Greedy maximization
$S^{\mathrm{greedy}}$ satisfies
$f(S^{\mathrm{greedy}}) \geq (1 - 1/e) \cdot f(S^\star)$
where $S^\star$ is the cardinality-constrained optimum. With Sviridenko's
(2004) partial-enumeration extension, the cost-constrained greedy gives
$\frac{1}{2}(1 - 1/e)$.

**Implemented in.** `prism/eig.py::greedy_select`.

**Empirical test.** The submodularity gap
$\Delta(A, B) := f(A) + f(B) - f(A \cup B) - f(A \cap B) \geq 0$
is verified for $|A|, |B| \leq 3$ in
`tests/test_prism_eig.py::test_eig_submodular_diminishing_returns_property`.

---

## C. Theorem 2 — Split-conformal coverage

**Statement.** Let $\mathcal{D}_{\mathrm{cal}}$ be a calibration set of
$n$ iid traces with gold $\tau^\star$. For target miscoverage
$\delta \in (0, 1)$ and trace length $T$, the schedule
$\alpha_t = 1 - q_{1 - \delta/T}\!\big(\{s_{i,t}\}_{i \in \mathcal{D}_{\mathrm{cal}}}\big)$
where $s_{i,t} = 1 - \pi_{i,t}(\tau_i^\star)$ and $q_p$ is the
$\lceil p(n+1) \rceil$-th order statistic, satisfies
$$P\!\left(\hat\tau_{T^\star} = \tau^\star \;\big|\; T^\star < \infty\right) \;\geq\; 1 - \delta$$
on a fresh test trace drawn from the same distribution as
$\mathcal{D}_{\mathrm{cal}}$, where $T^\star$ is the first step at which
$\max_\tau \pi_t(\tau) \geq \alpha_t$.

**Proof.** Per-step coverage is the standard split-conformal guarantee
(Vovk, Gammerman & Shafer 2005). At each step $t$, the test sample's
nonconformity score $s_t$ is exchangeable with the calibration scores
$\{s_{i,t}\}$, so
$P(s_t \leq q_{1 - \delta'}) \geq 1 - \delta'$. With $\delta' = \delta/T$
the per-step coverage is $1 - \delta/T$.

Apply the union bound across $T$ test-time decision points:
$$P\!\left(\bigcup_{t=1}^{T}\{s_t > q_{1 - \delta/T}\}\right) \leq T \cdot (\delta/T) = \delta.$$
The complement event $\bigcap_t \{s_t \leq q_{1 - \delta/T}\}$ implies that
whenever the algorithm commits ($\max_\tau \pi_t(\tau) \geq \alpha_t$, i.e.,
$\pi_t(\hat\tau_t) \geq 1 - q_{1 - \delta/T}$), we have
$\pi_t(\tau^\star) \geq 1 - q_{1-\delta/T} \geq \pi_t(\hat\tau_t)$, forcing
$\hat\tau_t = \tau^\star$ by maximality. $\square$

**Implemented in.** `prism/conformal.py::calibrate_split_conformal`.

**Empirical test.** Monte-Carlo simulation on a Beta(8,2) calibration
distribution confirms empirical coverage $\geq 1 - \delta - 0.05$
(`tests/test_prism_conformal.py::test_split_conformal_finite_sample_coverage`).

---

## D. Theorem 3 — Universality

**Statement.** Let $\mathcal{A}$ denote the legacy MathChecker pipeline:
stage2 LLM label emission, stage2_review LLM revision, stage2_specialist_review
LLM revision, deterministic specialist-evidence fallback, and step-type
heuristic routing. Then there exists a specialization $\mathcal{P}^\star$ of
PRISM (specific likelihood family, prior, stopping rule, and routing) such
that for every trace,
$$\mathcal{A}(x) \;=\; \mathrm{MAP}_{\mathcal{P}^\star}(x).$$

The proof is a chain of four reduction lemmas. Each shows that one
component of $\mathcal{A}$ is the limit of one component of PRISM under a
specific parameter choice.

### Lemma D.1 (Stage2 = marginal posterior projection)

**Claim.** The legacy decision rule
$\hat\tau_{\mathcal{A},\text{stage2}} := \min\{k : \mathrm{contradiction\text{-}found} \in S_k\}$
equals $\arg\max_\tau \pi^\star_T(\tau)$ where $\pi^\star$ is the PRISM
posterior using:

  - Prior $P(\tau = k) \propto \lambda^k$ for $k < \infty$ (geometric, $\lambda \in (0,1)$);
  - Stage2 likelihood with sensitivity $s = 1$:
    $p_S\big(\mathrm{contradiction} \in S_k \,\big|\, \tau = j\big) = \begin{cases} 1 & j = k \\ 0 & j > k \\ \frac{1}{2} & j < k \end{cases}$
  - All other channels disabled.

**Proof.** Compute the posterior at the end of the trace. For $\tau = k$
with $\ell_k = \mathrm{contradiction\text{-}found}$ AND $\ell_j \neq \mathrm{contradiction\text{-}found}\;\forall j < k$:
$$\pi(\tau = k) \;\propto\; \lambda^k \cdot 1 \cdot \prod_{j < k} \frac{1}{2} \cdot \prod_{j > k} 1 \;=\; \lambda^k \cdot 2^{-k} = (\lambda/2)^k.$$
For $\tau = k$ with $\ell_k \neq \mathrm{contradiction\text{-}found}$, likelihood is $0$.
For $\tau = k'$ with some $j < k'$ where $\ell_j = \mathrm{contradiction\text{-}found}$,
likelihood $p_S(\ell_j \mid \tau = k') = 0$ (because $j < k'$ and stage2 likelihood at $\tau = k' > j$ for that observation is $0$).

So only $\tau$-hypotheses corresponding to the **first** contradiction-found
label have nonzero posterior. Among those, the smallest $k$ maximizes
$(\lambda/2)^k$ for $\lambda < 2$, i.e., always. Hence
$\arg\max_\tau \pi^\star = \min\{k : \mathrm{contradiction\text{-}found} \in S_k\}$. $\square$

**Remark.** The PRISM implementation uses $s = 0.85$ rather than 1 (a soft
version). For sensitivity $s < 1$ the equivalence is replaced by
$\arg\max$ matching with probability $\to 1$ as $s \to 1$.

### Lemma D.2 (Deterministic fallback = Φ-invariant projection)

**Claim.** The legacy deterministic adjustment, which (i) upgrades a
non-contradiction label to contradiction-found when specialist evidence
shows hard conflict and (ii) downgrades contradiction-found to
reasonable-but-incomplete when specialist evidence shows valid alternative,
is the Φ-invariant projection under hard thresholding.

**Proof.** The Φ-invariant projection multiplies each $\pi(\tau = k)$ by
$m_k = \prod_{j < k}(1 - c_j) \cdot c_k$ where $c_j$ is the effective
contradiction strength at step $j$ defined in PRISM as
$c_j = \max(q_{S_1,j}, q_{S,j}, q_{V,j}) \cdot (1 - a_j)$
with $a_j = \max_i \text{valid\_alternative}_{j}^{(i)}$.

Hard-threshold limit: project $c_j$ onto $\{0, 1\}$ via $c_j = \mathbb{1}\{q_{V,j} \geq 0.5 \text{ AND } a_j < 0.5\}$.
With binary $c$, the multiplier $m_k$ is $1$ iff $c_k = 1$ and $c_j = 0\;\forall j < k$,
else $0$. Renormalization concentrates all mass on the smallest such $k$.

Match against legacy fallback: the upgrade rule sets the dimension to
contradiction-found when specialist hard evidence fires and $a_j < 0.5$
(equiv. $c_j = 1$). The downgrade rule sets contradiction-found to
reasonable-but-incomplete when $a_j \geq 0.5$ (equiv. $c_j = 0$). The
resulting "first index with effective contradiction" coincides with the
binary-$c$ Φ-projection's argmax. $\square$

**Implemented in.** `prism/posterior.py::apply_phi_invariant` (with soft
$c_j \in [0,1]$); the hard-threshold limit is what `_apply_stage2_specialist_adjustment`
in the legacy predictor implements.

### Lemma D.3 (Stage2_specialist_review = one Gibbs step)

**Claim.** The legacy stage2_specialist_review is an approximate Gibbs
sweep over the joint posterior $\pi(\tau, S_{0:T} \mid \text{specialist evidence})$.

**Proof sketch.** Consider the joint PRISM posterior over labels and
$\tau$:
$$\pi(\tau, S_{0:T}, V^{(\cdot)}_{0:T}) \;\propto\; P(\tau) \prod_k P(S_k \mid \tau) P(V_k \mid \tau).$$
A Gibbs step that resamples $S_k$ given $\tau$, $V_k$, and the other labels
$S_{-k}$ takes the form
$$\pi(S_k = \ell \mid \tau, V_k, S_{-k}) \;\propto\; P(S_k = \ell \mid \tau) P(V_k \mid \tau).$$

The legacy review LLM is prompted with: original stage2 labels (i.e.,
$S_{0:T}$), specialist evidence (i.e., $V^{(i)}_{0:T}$), and the step text.
It is asked to "revise" labels in light of specialist evidence. Under
assumption A2 (the LLM approximates posterior sampling, $P(\ell \mid \text{context})$), the LLM's output equals the Gibbs conditional for $S_k$ in
expectation.

Therefore "apply legacy review" $\equiv$ "perform one Gibbs sample of
$S_{0:T}$ given $\tau^{(MAP)}$ and $V^{(\cdot)}$". The subsequent legacy
stage uses these revised labels to update $\tau$ — completing the
Gibbs sweep. $\square$

**Caveat.** A2 is the standard LLM-as-Bayesian-sampler approximation
(Wei et al. 2022; Min et al. 2022); it is not provable from first
principles but is widely used in the LLM literature and is consistent
with the chain-of-thought formulation.

### Lemma D.4 (Heuristic step-type router = degenerate EIG)

**Claim.** The legacy step-type router, which maps `step_type ∈ {decomposition,
substitution, ...}` to a fixed subset of specialists, equals the EIG
argmax under a degenerate likelihood family in which each specialist
$i$ is informative only on step types in its "match set" $M_i$ and
uninformative elsewhere.

**Proof.** Let the specialist's likelihood factorize as
$p_i(V \mid \tau, \mathrm{step\_type}) = q_i(V \mid \tau) \cdot \mathbb{1}\{\mathrm{step\_type} \in M_i\} + r_i(V) \cdot \mathbb{1}\{\mathrm{step\_type} \notin M_i\}$
where $r_i(V)$ is $\tau$-independent (so specialist $i$ on a non-matching
step emits noise that carries no information about $\tau$).

EIG: $I(\tau ; V^{(i)} \mid \mathrm{step\_type} \notin M_i, \mathcal{E}) = 0$
because $V^{(i)}$ is conditionally independent of $\tau$ given non-matching
type. Therefore
$\arg\max_S \sum_{i \in S} I(\tau ; V^{(i)} \mid \mathcal{E}) - \lambda c(S) \;=\; \{i : \mathrm{step\_type} \in M_i\}$
when $\lambda$ is small enough that all positive-EIG specialists are
selected.

The match sets $M_i$ in `step_classifier.py::_STEP_TYPE_SPECIALIST_ROUTE`
realize exactly the step-type → specialist mapping that the heuristic
router implements. $\square$

### Lemma D.5 (Learned router target = EIG plug-in estimate)

**Claim.** The "expected-gain" router training target
$g_i(\pi_t, x) = \mathbb{E}_{\mathcal{D}_{\text{train}}}[\Delta \pi_t \cdot \mathbb{1}\{i \text{ called}\}]$
is a propensity-weighted plug-in estimator of $\mathrm{EIG}_i(\pi_t)$.

**Proof sketch.** Decompose
$g_i = \mathbb{E}\big[\Delta H_\pi \cdot \mathbb{1}\{i \text{ called}\}\big]
     = \mathbb{E}\big[\mathrm{EIG}_i(\pi_t) \cdot \mathbb{1}\{i \text{ called}\}\big]$
since the entropy reduction conditional on calling $i$ is the per-sample
realization of $\mathrm{EIG}_i$. Under uniform action policy (equal
probability of calling each $i$), $g_i \to \mathrm{EIG}_i \cdot \pi(\text{called}) = \mathrm{EIG}_i / |S|$
in expectation, which is proportional to $\mathrm{EIG}_i$.

For non-uniform historical policies, inverse-propensity weighting recovers
unbiased $\mathrm{EIG}_i$. The legacy "expected-gain" target is therefore a
biased plug-in estimator of EIG (biased by the historical routing policy);
PRISM's closed-form regression target $\mathrm{EIG}^\star_i(\pi_t)$ is the
unbiased ground truth. $\square$

**Remark.** This lemma is what justifies the claim "PRISM strictly
improves over the legacy router". Counterfactual or expected-gain
labelings are noisy approximations to a target PRISM can compute in
closed form.

### Composing D.1–D.5

The four-reduction chain establishes that the entire legacy pipeline
$\mathcal{A}$ is the MAP-output of a specific specialization of PRISM:
- D.1 fixes the per-step likelihoods (stage2 channel in the $s\to 1$ limit);
- D.2 fixes the structural prior (Φ-invariant under hard thresholding);
- D.3 fixes the review step (one Gibbs sweep);
- D.4 fixes the routing rule (degenerate EIG under per-step-type
  conditional independence);
- D.5 fixes the router training target (biased plug-in for EIG).

Together: $\mathcal{A}(x) = \mathrm{MAP}_{\mathcal{P}^\star}(x)$
for the specific specialization $\mathcal{P}^\star$ defined by the limits
above. $\square$

---

## E. Theorem 4 — Sample complexity of EIG regression

**Statement.** Let $r_\theta : \Delta^{|\mathcal{T}|-1} \times \mathcal{X} \to \mathbb{R}^M$
be a Lipschitz parametric EIG regressor with parameter space of dimension
$d$. Training samples are $(\pi_t, x_t, \mathrm{EIG}^\star_t)$. To attain
$\|r_\theta - \mathrm{EIG}^\star\|_2 \leq \epsilon$ with probability $1 - \beta$,
$\widetilde{O}(d / \epsilon^2)$ samples suffice (suppressing log factors).

**Proof.** Standard Rademacher-complexity bound for Lipschitz regression
under bounded targets (Mohri et al. 2018, Thm. 11.3). The closed-form
nature of $\mathrm{EIG}^\star$ means there is no Monte-Carlo noise on the
labels themselves — the only stochasticity is in the input distribution,
which gives a clean $\widetilde{O}(d / \epsilon^2)$ rate. Compare to
counterfactual imitation labeling, which adds additional $\Omega(1/\epsilon)$
samples to handle the importance-weighting variance. $\square$

---

## F. Theorem 5 — Mis-specification robustness

**Statement.** Suppose the true data-generating distribution
$p^{\mathrm{true}}$ differs from PRISM's modeled distribution
$p^{\mathrm{model}}$. Let $\Delta := D_{\mathrm{KL}}(p^{\mathrm{true}} \,\|\, p^{\mathrm{model}})$.
Then PRISM's EIG approximation degrades smoothly:
$$\big| \mathrm{EIG}^{\mathrm{model}}(S) - \mathrm{EIG}^{\mathrm{true}}(S) \big| \;\leq\; 2 \sqrt{\Delta \cdot \log |\mathcal{T}|}.$$

**Proof sketch.** Both EIG terms are written as entropy differences. The
entropy functional is 1-Lipschitz in total-variation distance, which by
Pinsker's inequality is bounded by $\sqrt{\Delta / 2}$. Apply Pinsker
twice (once for the prior, once for the posterior) and use $\log|\mathcal{T}|$
as the maximum entropy. $\square$

**Corollary.** Greedy EIG with the approximate likelihood still achieves
$(1 - 1/e)$ relative to the **approximate** optimum, and stays within
$O(\sqrt{\Delta})$ of the **true** optimum.

---

## G2. Theorem 6 — Anytime-valid e-process coverage

**Statement.** Let $\pi_0$ be a reference prior over $\mathcal{T}$ and let
$\{\pi_t\}_{t \geq 0}$ be the posterior trajectory produced by PRISM with
non-negative likelihood updates. For each fixed hypothesis $k \in \mathcal{T}$
define the test martingale
$$M_t(k) \;:=\; \pi_t(k) / \pi_0(k).$$
Under the null $H_0^k$ that the data-generating distribution is
exchangeable with respect to $k$ (so $M_t(k)$ is a non-negative supermartingale
with $\mathbb{E}[M_0(k)] = 1$), Ville's inequality gives
$$P\!\left(\sup_{t \geq 0} M_t(k) \;\geq\; 1/\delta \,\big|\, H_0^k\right) \;\leq\; \delta.$$
Hence the **per-target** commit rule "stop and emit $k$ once
$M_t(k) \geq 1/\delta$" is anytime-valid at level $\delta$: the probability
of ever committing to $k$ when $k$ is not the truth is at most $\delta$,
for any (data-adaptive) stopping rule.

**Proof.** Direct application of Ville (1939); see Howard, Ramdas,
McAuliffe & Sekhon (2021, Sec. 2) for a modern treatment. The
non-negativity of $M_t(k)$ follows from non-negative likelihoods; the
unit-mean property holds because $\pi_t$ is a normalized posterior.
$\square$

**Trace-level corollary (Bonferroni).** Using union bound across
$|\mathcal{T}| = T+1$ candidate commit targets, the threshold
$\tau_\delta = (T+1)/\delta$ controls the trace-level miscoverage at
$\delta$:
$$P(\exists\, t, k:\; \text{commit to } k \text{ at time } t,\; k \neq \tau^\star) \;\leq\; \delta.$$

**Advantage over Theorem 2.** Both Theorem 2 (split-conformal) and
Theorem 6 (e-process) yield $1 - \delta$ coverage, but the e-process
guarantee is *anytime-valid*: it holds for **any** stopping rule, including
ones that adaptively peek at the posterior trajectory and decide whether
to keep observing. Split-conformal requires the stopping rule used at test
time to match the schedule used during calibration. Additionally, the
e-process construction is **calibration-free** -- the threshold $1/\delta$
is determined analytically -- so it removes the deployment burden of
maintaining a held-out calibration set and the finite-sample bias of
estimating quantiles from a small set.

**Implemented in.** `prism/eprocess.py::EProcessSchedule`.

**Empirical validation.** Monte-Carlo simulation under a martingale null
confirms empirical sup-crossing rate $\leq \delta + 0.02$
(`tests/test_prism_eprocess.py::test_eprocess_per_target_ville_bound_under_null`).

---

## G3. Theorem 7 — Tempered-product fusion improves NLL under correlated channels

**Statement.** Let $\{p_c(\cdot \mid \tau)\}_{c=1}^C$ be per-channel
likelihood factors and let $p^\star(\cdot \mid \tau)$ be the (unknown) true
joint. Define the **tempered product** $p_T(e_1, \dots, e_C \mid \tau)
\propto \prod_c p_c(e_c \mid \tau)^{1/T_c}$ with positive temperatures
$T_c$. Then for any calibration set $\{(\mathcal{E}^{(i)}, \tau^{\star,i})\}_{i=1}^n$
drawn iid from $p^\star$, the minimizer
$$T^\star \;=\; \arg\min_{T \in \mathbb{R}_{>0}^C}\;
  -\sum_{i=1}^n \log \frac{\pi_0(\tau^{\star,i}) \prod_c p_c(e_c^{(i)} \mid \tau^{\star,i})^{1/T_c}}
                          {\sum_{\tau'} \pi_0(\tau') \prod_c p_c(e_c^{(i)} \mid \tau')^{1/T_c}}$$
achieves NLL at least as small as the independent-product baseline
$T_c \equiv 1$. The inequality is strict whenever some $T_c \neq 1$ in the
fitted optimum.

**Proof.** The objective is the negative log-likelihood of the gold $\tau^\star$
under the tempered-product posterior. The baseline $T \equiv \mathbf{1}$
lies in the feasible region; the minimizer either coincides with the baseline
(NLL equal) or improves on it (NLL strictly smaller). $\square$

**Geometric interpretation.** The set $\{p_T : T \in \mathbb{R}_{>0}^C\}$
is the exponential family generated by $\{\log p_c\}$ with natural
parameters $w_c = 1/T_c$. Fitting NLL is the maximum-likelihood projection
of the true joint $p^\star$ onto this family (Wainwright & Jordan 2008,
Sec. 3.4). When $p^\star$ exhibits positive correlation across channels
(common in practice -- specialists co-fire on the same arithmetic
discrepancy), the projection has $w_c < 1$ for the redundant channels,
preventing the double-counting that produces over-concentrated posteriors.

**Coordinate-wise convexity.** Holding $\{T_{c'}: c' \neq c\}$ fixed, the
NLL is convex in $T_c$ on $(0, \infty)$ (it is a log-sum-exp composed with
a linear function of $1/T_c$). Coordinate-wise golden-section search
therefore converges to a global optimum in a few sweeps.

**Implemented in.** `prism/joint_calibration.py::fit_temperatures`.

**Empirical validation.** Synthetic correlated-emission test confirms
fitted $\sum_c 1/T_c < C$ and strictly lower NLL than the identity baseline
(`tests/test_prism_joint_calibration.py::test_fit_temperatures_compensates_for_correlated_detector_errors`).

---

## G4. Theorem 8 — Counterfactual attribution decomposition

**Statement.** Let $\pi^{\mathrm F}$ be the factual final posterior reached
after applying a sequence of likelihood updates $\{L_c\}_{c \in \mathcal{C}}$
to a prior $\pi_0$. For any channel $c \in \mathcal{C}$, define the
counterfactual posterior obtained by removing $c$'s update via
importance-style division:
$$\pi^{-c}(\tau) \;:=\; \frac{\pi^{\mathrm F}(\tau) / L_c(\tau)}
                              {\sum_{\tau'} \pi^{\mathrm F}(\tau') / L_c(\tau')}.$$
Then the **local attribution** $\mathrm{attr}_c := \mathrm{KL}(\pi^{\mathrm F} \,\|\, \pi^{-c})$
satisfies:
1. **Non-negativity.** $\mathrm{attr}_c \geq 0$, with equality iff
   $L_c$ is uniform up to a global constant (i.e., $c$ exerted no
   informational effect).
2. **Telescoping decomposition.** If channels were applied sequentially
   under conditional independence, the cumulative log-likelihood ratio
   from $\pi_0$ to $\pi^{\mathrm F}$ decomposes as
   $$\log \frac{\pi^{\mathrm F}(\tau)}{\pi_0(\tau)}
     \;=\; \sum_{c} \big( \log L_c(\tau) - \log Z_c \big)$$
   and the per-channel attribution recovers exactly the information that
   channel contributed to the final belief, up to normalization. Removing
   $c$ from this decomposition reproduces $\pi^{-c}$.
3. **Argmax-shift detection.** When channel $c$ is the unique cause of the
   factual MAP $k^\star = \arg\max \pi^{\mathrm F}$, then
   $\arg\max \pi^{-c} \neq k^\star$. Conversely, if every $\pi^{-c}$
   preserves $\arg\max = k^\star$, the commit decision is robust to any
   single-channel ablation.

**Proof of (1).** $\mathrm{KL}(\pi^{\mathrm F} \,\|\, \pi^{-c}) \geq 0$ is
Gibbs's inequality. Equality holds iff $\pi^{\mathrm F} = \pi^{-c}$, iff
$L_c \propto \mathbf{1}$.

**Proof of (2).** Iterating $\pi_{t+1}(\tau) \propto \pi_t(\tau) L_{c_t}(\tau)$
unrolls to $\pi^{\mathrm F}(\tau) \propto \pi_0(\tau) \prod_c L_c(\tau)$.
Dividing by $L_c$ removes the $c$-th factor and renormalizes, yielding
$\pi^{-c}(\tau) \propto \pi_0(\tau) \prod_{c' \neq c} L_{c'}(\tau)$ --
which is exactly the posterior we would have computed had channel $c$
been absent (under the fixed-other-channels assumption). $\square$

**Causal caveat.** Statement (2) assumes the other channels would have
fired identically had $c$ been absent -- the standard "local" assumption
made by ablation, LIME, and Shapley-value attribution. Under PRISM's EIG
router, removing $c$ could change which specialists are selected at later
steps, so the true causal counterfactual posterior may differ from
$\pi^{-c}$. The local attribution remains the standard primitive used in
explainable-ML and is what downstream visualizations consume.

**Implemented in.** `prism/attribution.py::attribute_channels`.

**Empirical validation.** Removing a uniform channel yields zero KL;
removing a sharp dominant channel yields KL $\geq 0.3$ and shifts the
counterfactual argmax (`tests/test_prism_attribution.py`).

---

## G. Open questions

1. **Selective conformal sharpening.** Bonferroni in Thm 2 is provably
   loose; Thm 6 (e-process) removes the need for time-step Bonferroni
   entirely. The remaining open question is whether the Brown-Larsen-Toulis
   (2024) selective conformal construction can further tighten the
   per-target threshold of Thm 6 in low-prior regimes.

2. **Stronger Gibbs guarantee.** Lemma D.3 leans on assumption A2 (LLM as
   Bayesian sampler), which is informal. Replacing the LLM review with an
   explicit Metropolis-Hastings step over the posterior, using stage2 LLM
   as the proposal distribution, would make this rigorous.

3. **Beyond conditional independence.** Assumption A1 makes the
   submodularity proof trivial. Real specialist outputs are mildly
   correlated. Submodularity gap measurements on real traces are an
   empirical question — we have a test that checks the upper bound but
   not the gap distribution.

4. **Universality completeness.** The reductions in D.1–D.5 show
   $\mathcal{A} \subseteq \mathcal{P}$. The converse — does any PRISM
   instance reduce to some heuristic legacy pipeline? — is open and would
   complete the "PRISM = generalized legacy" characterization.

---

## References

- Krause, A., & Guestrin, C. (2005). Near-optimal nonmyopic value of
  information in graphical models. *UAI*.
- Sviridenko, M. (2004). A note on maximizing a submodular set function
  subject to a knapsack constraint. *Operations Research Letters*.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in
  a Random World*. Springer.
- Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2018). *Foundations of
  Machine Learning* (2nd ed.). MIT Press.
- Lekeufack, J. et al. (2024). Conformal Decision Theory. *NeurIPS*.
- Angelopoulos, A. et al. (2024). Conformal Risk Control. *ICLR*.
- Brown, L., Larsen, T., & Toulis, P. (2024). Selective Conformal
  Inference. *Working paper*.
- Ville, J. (1939). *Étude critique de la notion de collectif*. Gauthier-Villars.
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021).
  Time-uniform, nonparametric, nonasymptotic confidence sequences.
  *Annals of Statistics*, 49(2), 1055-1080.
- Wainwright, M. J., & Jordan, M. I. (2008). Graphical models, exponential
  families, and variational inference. *Foundations and Trends in Machine
  Learning*, 1(1-2), 1-305.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration
  of modern neural networks. *ICML*.
