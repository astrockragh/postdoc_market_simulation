"""
ApplicantCohort — postdoctoral job market simulation
=====================================================

Round structure
---------------
Each offer round consists of two phases:

  OFFER PHASE
    Every open job walks down its ranked pool and makes an offer to the first
    applicant it finds who is still on the market.  If the pool is exhausted,
    the job triggers a "second call": a fresh mini-pool is sampled from all
    remaining market participants (weighted by skill via softmax), each
    selected applicant's n_applications is incremented by 1, the mini-pool is
    ranked with a fresh Mallows draw, and the job continues from there.
    Second calls can repeat as many times as needed until the job fills or the
    market is empty.

  ACCEPT PHASE
    Every applicant who received at least one offer this round picks the best
    one (highest prestige * lognormal perturbation), accepts it, and leaves
    the market.  All remaining offers are implicitly declined.

Key design choices
------------------
- n_applications tracks how many pools an applicant entered (original + each
  second call they were drawn into).  It is NOT a cap on offers received.
- self.market is a Python set for O(1) membership test and removal.
- Jobs and applicants are stored as plain dicts — easy to inspect.
- Verbose levels:
    0  silent
    1  round-level summaries + market open/close
    2  every offer event + second-call triggers
    3  full debug: pool walks, skip reasons, second-call sampling details

Inject a test candidate
-----------------------
    cohort = ApplicantCohort(verbose=2)
    cohort.rank_applicants()
    cohort.inject_candidate(student_percentile=10, n_applications=15)
    cohort.run_market()
    result = cohort.get_injected_result()
"""

import sys, os
import numpy as np
from collections import defaultdict
from typing import Optional, List, Tuple


# =============================================================================
# Helpers
# =============================================================================

def _mallows_rank(pool: list, applicants: list, phi: float) -> list:
    """
    Return a Mallows-perturbed ranking of `pool` (list of applicant indices).
    The reference (modal) ranking is skill-descending order.
    phi=0 -> pure skill order, phi=1 -> uniform random.
    """
    sys.path.append(os.path.join(os.path.dirname(__file__), "top-k-mallows"))
    import mallows_kendall as mk

    if len(pool) == 0:
        return []
    if len(pool) == 1:
        return list(pool)

    # Reference = skill descending
    ref = sorted(pool, key=lambda i: applicants[i]["skill"], reverse=True)
    perm = mk.sample(m=1, n=len(ref), phi=phi)[0]
    return [ref[p] for p in perm]


def _softmax_sample(candidates: list, applicants: list,
                    n_draw: int, temperature: float) -> list:
    """
    Sample n_draw items from candidates without replacement,
    weighted by softmax(skill / temperature).
    """
    n_draw = min(n_draw, len(candidates))
    skills = np.array([applicants[i]["skill"] for i in candidates], dtype=float)
    logits = skills / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    idx = np.random.choice(len(candidates), size=n_draw, replace=False, p=probs)
    return [candidates[i] for i in idx]


# =============================================================================
# ApplicantCohort
# =============================================================================

class ApplicantCohort:

    def __init__(
        self,
        n_students                = 1000,
        n_postdocs                = 300,
        splits                    = np.array([6, 50, 200]),
        frac_student_applying     = 3 / 4,
        frac_postdoc_applying     = 1 / 3,
        mean_applications         = np.log(30),
        sigma_applications        = 0.4,
        tier_max_prestige         = [12, 15, 3],
        prestige_decays           = [1, 2, 1.2],
        minimum_prestiges         = [5, 3, 1],
        loc_postdoc_reapplying    = 1.3,
        scale_postdoc_reapplying  = 0.8,
        loc_student               = 0,
        scale_student             = 1,
        self_awareness            = 0.5,
        stochasticity             = 0.7,
        second_call_loc           = 80,
        second_call_scale         = 15,
        second_call_min           = 40,
        second_call_temperature   = 1.0,
        indecision_noise          = 0.2,
        verbose                   = 1,
    ):
        self.n_students               = n_students
        self.n_postdocs               = n_postdocs
        self.frac_student_applying    = frac_student_applying
        self.frac_postdoc_applying    = frac_postdoc_applying
        self.n_student_applying       = int(n_students * frac_student_applying)
        self.n_postdoc_reapplying     = int(n_postdocs * frac_postdoc_applying)
        self.splits                   = np.array([int(s) for s in splits])
        self.n_positions              = int(sum(self.splits))
        self.tier_max                 = np.array(tier_max_prestige)
        self.prestige_decays          = list(prestige_decays)
        self.minimum_prestiges        = list(minimum_prestiges)
        self.mean_applications        = mean_applications
        self.sigma_applications       = sigma_applications
        self.self_awareness           = float(np.clip(self_awareness, 0, 0.5))
        self.indecision_noise         = indecision_noise
        self.stochasticity            = stochasticity
        self.loc_postdoc_reapplying   = loc_postdoc_reapplying
        self.scale_postdoc_reapplying = scale_postdoc_reapplying
        self.loc_student              = loc_student
        self.scale_student            = scale_student
        self.second_call_loc          = second_call_loc
        self.second_call_scale        = second_call_scale
        self.second_call_min          = second_call_min
        self.second_call_temperature  = second_call_temperature
        self.verbose                  = verbose

        self._injected_idx   = None   # set by inject_candidate()
        self._injected_batch = []     # set by inject_candidates_batch(): list of
                                      # {"idx": int, "student_percentile": float,
                                      #  "n_applications": int}

        self.applicants = []        # list of dicts
        self.jobs       = []        # list of dicts
        self.market     = set()     # indices of applicants still on market

    # -------------------------------------------------------------------------
    # STEP 1 — rank applicants
    # -------------------------------------------------------------------------

    def rank_applicants(self):
        """Draw skills, merge students + postdocs, sort best-first."""

        skills_s = np.random.normal(
            self.loc_student, self.scale_student, self.n_student_applying)
        skills_p = np.random.normal(
            self.loc_postdoc_reapplying, self.scale_postdoc_reapplying,
            self.n_postdoc_reapplying)

        all_skills = np.concatenate([skills_s, skills_p])
        all_labels = (["student"] * self.n_student_applying
                      + ["postdoc"] * self.n_postdoc_reapplying)

        order = np.argsort(-all_skills)   # best first

        self.applicants = []
        for rank, orig in enumerate(order):
            self.applicants.append({
                "id":                rank,
                "rank":              rank,
                "skill":             float(all_skills[orig]),
                "currently":         all_labels[orig],
                "injected":          False,
                # application tracking
                "n_applications":    0,       # filled in assign_application_counts
                "pools_entered":     [],      # job ids (original + second call)
                # offer tracking (reset each round)
                "offers_this_round": [],      # list of (job_id, prestige)
                # full history
                "all_offers":        [],      # list of dicts
                # outcome
                "accepted":          False,
                "accepted_job":      None,
                "accepted_prestige": None,
                "accepted_round":    None,
            })

        self.market = set(range(len(self.applicants)))

        if self.verbose >= 1:
            n_s = sum(1 for a in self.applicants if a["currently"] == "student")
            n_p = sum(1 for a in self.applicants if a["currently"] == "postdoc")
            print(f"\n[rank_applicants]  {len(self.applicants)} applicants"
                  f"  ({n_s} students, {n_p} postdocs)")
            print(f"  skill range: [{self.applicants[-1]['skill']:.3f}, "
                  f"{self.applicants[0]['skill']:.3f}]")

    # -------------------------------------------------------------------------
    # INJECT CANDIDATE  (optional — between rank_applicants and run_market)
    # -------------------------------------------------------------------------

    def inject_candidate(self, student_percentile: float, n_applications: int):
        """
        Insert a test candidate at a fixed student-percentile with a fixed
        application count.  Call after rank_applicants(), before run_market().

        student_percentile : 0 = best student, 100 = worst student.
        """
        if not self.applicants:
            raise RuntimeError("Call rank_applicants() first.")

        student_skills = np.array([
            a["skill"] for a in self.applicants if a["currently"] == "student"
        ])
        target_skill = float(np.percentile(student_skills, 100 - student_percentile))

        candidate = {
            "id":                -1,
            "rank":              -1,
            "skill":             target_skill,
            "currently":         "student",
            "injected":          True,
            "n_applications":    int(n_applications),
            "pools_entered":     [],
            "offers_this_round": [],
            "all_offers":        [],
            "accepted":          False,
            "accepted_job":      None,
            "accepted_prestige": None,
            "accepted_round":    None,
        }

        # Insert keeping skill-descending order
        insert_pos = len(self.applicants)
        for i, a in enumerate(self.applicants):
            if target_skill >= a["skill"]:
                insert_pos = i
                break

        self.applicants.insert(insert_pos, candidate)

        # Re-index everything
        for i, a in enumerate(self.applicants):
            a["id"]   = i
            a["rank"] = i

        self.market = set(range(len(self.applicants)))
        self._injected_idx = insert_pos

        if self.verbose >= 1:
            n_total   = len(self.applicants)
            ovr_pct   = insert_pos / n_total * 100
            stu_list  = [a for a in self.applicants if a["currently"] == "student"]
            stu_rank  = sum(1 for a in self.applicants[:insert_pos]
                            if a["currently"] == "student")
            print(f"\n[inject_candidate]")
            print(f"  student_percentile : {student_percentile:.1f}%")
            print(f"  target_skill       : {target_skill:.4f}")
            print(f"  overall position   : {insert_pos} / {n_total}"
                  f"  ({ovr_pct:.1f}% overall)")
            print(f"  student rank       : {stu_rank} / {len(stu_list)}")
            print(f"  n_applications     : {n_applications}  (fixed)")

    # -------------------------------------------------------------------------
    # INJECT CANDIDATES — BATCH  (for grid sweeps)
    # -------------------------------------------------------------------------

    def inject_candidates_batch(
        self,
        param_list: list,
    ):
        """
        Inject multiple test candidates in one call.  Each candidate is
        independent: they enter their own pools, receive their own offers,
        and are tracked separately.

        Must be called after rank_applicants(), before run_market().
        Cannot be combined with inject_candidate().

        Parameters
        ----------
        param_list : list of (student_percentile, n_applications) tuples
            e.g. [(10, 15), (25, 20), (50, 30)]

        After the call, self._injected_batch is populated with one entry
        per candidate:
            {"idx": int, "student_percentile": float, "n_applications": int}
        """
        if not self.applicants:
            raise RuntimeError("Call rank_applicants() first.")
        if self._injected_idx is not None:
            raise RuntimeError("Cannot mix inject_candidate and inject_candidates_batch.")

        # Pre-compute student skill distribution once
        student_skills = np.array([
            a["skill"] for a in self.applicants if a["currently"] == "student"
        ])

        # Build candidate dicts with target skills
        candidates_to_insert = []
        for pct, n_apps in param_list:
            target_skill = float(np.percentile(student_skills, 100 - pct))
            candidates_to_insert.append({
                "id":                -1,
                "rank":              -1,
                "skill":             target_skill,
                "currently":         "student",
                "injected":          True,
                "injected_pct":      float(pct),
                "injected_n_apps":   int(n_apps),
                "n_applications":    int(n_apps),
                "pools_entered":     [],
                "offers_this_round": [],
                "all_offers":        [],
                "accepted":          False,
                "accepted_job":      None,
                "accepted_prestige": None,
                "accepted_round":    None,
            })

        # Insert all candidates maintaining skill-descending order.
        # We insert highest-skill first so earlier insertions don't shift
        # the intended positions of later ones.
        for cand in sorted(candidates_to_insert,
                           key=lambda c: c["skill"], reverse=True):
            insert_pos = len(self.applicants)
            for i, a in enumerate(self.applicants):
                if cand["skill"] >= a["skill"]:
                    insert_pos = i
                    break
            self.applicants.insert(insert_pos, cand)

        # Re-index all applicants
        for i, a in enumerate(self.applicants):
            a["id"]   = i
            a["rank"] = i

        self.market = set(range(len(self.applicants)))

        # Record final indices for tracking
        self._injected_batch = [
            {
                "idx":                a["id"],
                "student_percentile": a["injected_pct"],
                "n_applications":     a["injected_n_apps"],
            }
            for a in self.applicants if a.get("injected")
        ]

        if self.verbose >= 1:
            print(f"\n[inject_candidates_batch]  {len(self._injected_batch)} candidates injected")
        if self.verbose >= 2:
            for entry in self._injected_batch:
                a       = self.applicants[entry["idx"]]
                ovr_pct = entry["idx"] / len(self.applicants) * 100
                print(f"  pct={entry['student_percentile']:.0f}%  "
                      f"n_apps={entry['n_applications']}  "
                      f"skill={a['skill']:.3f}  "
                      f"overall_pos={entry['idx']} ({ovr_pct:.1f}%)")

    def get_all_injected_results(self) -> list:
        """
        Return a list of result dicts, one per injected candidate.
        Each dict includes the original (student_percentile, n_applications)
        parameters so results can be grouped by cell.
        """
        out = []
        for entry in self._injected_batch:
            a = self.applicants[entry["idx"]]
            out.append({
                "student_percentile":   entry["student_percentile"],
                "n_applications":       entry["n_applications"],
                "got_offer":            len(a["all_offers"]) > 0,
                "n_offers":             len(a["all_offers"]),
                "n_second_call_offers": sum(1 for o in a["all_offers"]
                                            if o["second_call"]),
                "accepted":             a["accepted"],
                "accepted_round":       a["accepted_round"],
                "accepted_prestige":    a["accepted_prestige"],
                "overall_percentile":   entry["idx"] / len(self.applicants) * 100,
            })
        return out



    def assign_application_counts(self):
        """Lognormal draw for all non-injected applicants."""
        mean_postdoc = np.log(np.exp(self.mean_applications) / 2)

        for a in self.applicants:
            if a["injected"]:
                continue
            if a["currently"] == "postdoc":
                n = np.random.lognormal(mean_postdoc, self.sigma_applications)
            else:
                n = np.random.lognormal(self.mean_applications, self.sigma_applications)
            a["n_applications"] = max(1, int(n))

        if self.verbose >= 1:
            stu = [a for a in self.applicants if a["currently"] == "student"]
            pdc = [a for a in self.applicants if a["currently"] == "postdoc"]
            print(f"\n[assign_application_counts]")
            print(f"  students: n={len(stu)}, "
                  f"mean_apps={np.mean([a['n_applications'] for a in stu]):.2f}, "
                  f"max={max(a['n_applications'] for a in stu)}")
            print(f"  postdocs: n={len(pdc)}, "
                  f"mean_apps={np.mean([a['n_applications'] for a in pdc]):.2f}, "
                  f"max={max(a['n_applications'] for a in pdc)}")

    # -------------------------------------------------------------------------
    # STEP 3 — build prestige
    # -------------------------------------------------------------------------

    def build_prestige(self):
        """Exponential prestige decay within each tier."""
        prestige_vals = []
        for n, t_max, t_decay, t_min in zip(
            self.splits, self.tier_max,
            self.prestige_decays, self.minimum_prestiges
        ):
            ranks = np.arange(n)
            prestige_vals.extend(t_max * np.exp(-t_decay / n * ranks) + t_min)

        prestige_arr = np.array(prestige_vals) / 2.75

        self.jobs = []
        for j, p in enumerate(prestige_arr):
            self.jobs.append({
                "id":            j,
                "prestige":      float(p),
                "filled":        False,
                "pool":          [],   # current active ranked pool
                "pointer":       0,    # next position to offer from
                "second_call_n": 0,    # cumulative second calls for this job
            })

        if self.verbose >= 1:
            print(f"\n[build_prestige]  {len(self.jobs)} positions")
            print(f"  prestige range: [{prestige_arr.min():.3f}, "
                  f"{prestige_arr.max():.3f}]")
            for t_idx, (n, label) in enumerate(
                zip(self.splits, [f"tier {i}" for i in range(len(self.splits))])
            ):
                tier_pres = prestige_arr[sum(self.splits[:t_idx]):
                                         sum(self.splits[:t_idx]) + n]
                print(f"  {label}: {n} positions, "
                      f"prestige [{tier_pres.min():.2f}, {tier_pres.max():.2f}]")

    # -------------------------------------------------------------------------
    # STEP 4 — build application pools
    # -------------------------------------------------------------------------

    def build_application_pools(self):
        """
        Each applicant selects n_applications jobs (skill-aware softmax over
        prestige), enters those pools, then each pool is Mallows-ranked.
        """
        N = len(self.jobs)
        max_skill = max(abs(a["skill"]) for a in self.applicants)

        for a in self.applicants:
            beta   = 0.5 - self.self_awareness * a["skill"] / max_skill
            logits = beta * np.array([j["prestige"] for j in self.jobs])
            logits -= logits.max()
            probs  = np.exp(logits)
            probs /= probs.sum()

            k = min(a["n_applications"], N)
            chosen = np.random.choice(N, size=k, replace=False, p=probs)

            for job_id in chosen:
                self.jobs[job_id]["pool"].append(a["id"])
                a["pools_entered"].append(int(job_id))

        # Mallows-rank each pool
        for j in self.jobs:
            j["pool"]    = _mallows_rank(j["pool"], self.applicants,
                                         self.stochasticity)
            j["pointer"] = 0

        if self.verbose >= 1:
            pool_sizes = [len(j["pool"]) for j in self.jobs]
            print(f"\n[build_application_pools]")
            print(f"  pool sizes: min={min(pool_sizes)}, "
                  f"mean={np.mean(pool_sizes):.1f}, "
                  f"max={max(pool_sizes)}, "
                  f"empty={sum(1 for s in pool_sizes if s == 0)}")
            self.pool_sizes = pool_sizes
        if self.verbose >= 3:
            for j in self.jobs:
                top = j["pool"][:3]
                skills = [f"{self.applicants[i]['skill']:.2f}" for i in top]
                print(f"  job {j['id']:3d} prestige={j['prestige']:.3f}: "
                      f"pool={len(j['pool'])}, top-3 skills={skills}")

    # -------------------------------------------------------------------------
    # SECOND CALL
    # -------------------------------------------------------------------------

    def _trigger_second_call(self, job: dict, round_num: int):
        """
        Build a fresh mini-pool from all remaining market participants,
        Mallows-rank it, and reset job pool/pointer.
        Each drawn applicant gets n_applications += 1.
        """
        remaining = list(self.market)
        if not remaining:
            if self.verbose >= 2:
                print(f"    [2nd-call] job {job['id']}: "
                      f"market empty — cannot trigger second call")
            return

        job["second_call_n"] += 1
        sc_n = job["second_call_n"]

        n_raw  = int(np.random.normal(self.second_call_loc, self.second_call_scale))
        n_draw = max(self.second_call_min, min(n_raw, len(remaining)))

        mini_pool = _softmax_sample(
            remaining, self.applicants, n_draw, self.second_call_temperature
        )

        for aid in mini_pool:
            self.applicants[aid]["n_applications"] += 1
            self.applicants[aid]["pools_entered"].append(job["id"])

        job["pool"]    = _mallows_rank(mini_pool, self.applicants, self.stochasticity)
        job["pointer"] = 0

        if self.verbose >= 2:
            injected_in_pool = any(
                self.applicants[i]["injected"] for i in mini_pool
            )
            print(f"    [2nd-call #{sc_n}] job {job['id']} "
                  f"(prestige={job['prestige']:.3f}): "
                  f"drew {n_draw} / {len(remaining)} remaining "
                  f"(raw={n_raw})"
                  f"{'  *** injected candidate in pool ***' if injected_in_pool else ''}")

        if self.verbose >= 3:
            top = job["pool"][:5]
            skills = [f"{self.applicants[i]['skill']:.2f}" for i in top]
            print(f"      mini-pool top-5 skills: {skills}")

    # -------------------------------------------------------------------------
    # OFFER PHASE
    # -------------------------------------------------------------------------

    def _offer_phase(self, round_num: int):
        """
        Each open job finds the first available applicant in its pool and
        makes one offer.  Triggers second calls inline when pool exhausted.
        """
        open_jobs = [j for j in self.jobs if not j["filled"]]

        if self.verbose >= 2:
            print(f"\n  [offer phase]  {len(open_jobs)} open jobs")

        for job in open_jobs:

            offer_made = False

            while not offer_made:

                # Walk the pool from current pointer
                pool_exhausted = True
                while job["pointer"] < len(job["pool"]):
                    aid = job["pool"][job["pointer"]]
                    job["pointer"] += 1

                    if aid not in self.market:
                        if self.verbose >= 3:
                            a = self.applicants[aid]
                            print(f"    job {job['id']}: skip {aid} "
                                  f"(skill={a['skill']:.3f}) — off market")
                        continue

                    # Offer
                    a = self.applicants[aid]
                    a["offers_this_round"].append((job["id"], job["prestige"]))
                    a["all_offers"].append({
                        "job_id":      job["id"],
                        "prestige":    job["prestige"],
                        "round":       round_num,
                        "second_call": job["second_call_n"] > 0,
                    })
                    offer_made    = True
                    pool_exhausted = False

                    if self.verbose >= 2:
                        sc_tag = (f" [2nd-call #{job['second_call_n']}]"
                                  if job["second_call_n"] > 0 else "")
                        inj_tag = "  *** INJECTED ***" if a["injected"] else ""
                        print(f"    job {job['id']}{sc_tag} "
                              f"prestige={job['prestige']:.3f}  ->  "
                              f"offer to {aid} "
                              f"skill={a['skill']:.3f} "
                              f"{a['currently']}{inj_tag}")
                    break

                if pool_exhausted:
                    if not self.market:
                        if self.verbose >= 2:
                            print(f"    job {job['id']}: pool exhausted + "
                                  f"market empty — stays open")
                        break
                    # Trigger a second call and loop back
                    self._trigger_second_call(job, round_num)

    # -------------------------------------------------------------------------
    # ACCEPT PHASE
    # -------------------------------------------------------------------------

    def _accept_phase(self, round_num: int) -> list:
        """
        Every applicant with >=1 offer picks the best one and accepts.
        Returns list of accepted applicant ids.
        """
        accepted_ids = []

        for aid in sorted(self.market):   # sorted for deterministic debug output
            a = self.applicants[aid]
            if not a["offers_this_round"]:
                continue

            n_off    = len(a["offers_this_round"])
            perturb  = np.abs(np.random.normal(1.0, self.indecision_noise, n_off))
            weighted = np.array([p for _, p in a["offers_this_round"]]) * perturb
            best_i   = int(np.argmax(weighted))
            best_job, best_pres = a["offers_this_round"][best_i]

            a["accepted"]          = True
            a["accepted_job"]      = best_job
            a["accepted_prestige"] = best_pres
            a["accepted_round"]    = round_num

            self.jobs[best_job]["filled"] = True
            accepted_ids.append(aid)

            if self.verbose >= 2:
                declined = [jid for jid, _ in a["offers_this_round"]
                            if jid != best_job]
                inj_tag  = "  *** INJECTED ***" if a["injected"] else ""
                print(f"    accept: {aid} skill={a['skill']:.3f} "
                      f"{a['currently']}{inj_tag}  "
                      f"->  job {best_job} prestige={best_pres:.3f}"
                      f"  ({n_off} offer(s)"
                      f"{', declined ' + str(declined) if declined else ''})")

        # Remove accepted applicants from market
        for aid in accepted_ids:
            self.market.discard(aid)

        # Clear round offers for everyone
        for a in self.applicants:
            a["offers_this_round"] = []

        if self.verbose >= 2:
            print(f"\n  [accept phase]  {len(accepted_ids)} accepted")

        return accepted_ids

    # -------------------------------------------------------------------------
    # ROUND SUMMARY
    # -------------------------------------------------------------------------

    def _print_round_summary(self, round_num: int, accepted_ids: list):
        if self.verbose < 1:
            return

        acc = [self.applicants[i] for i in accepted_ids]
        n_s = sum(1 for a in acc if a["currently"] == "student")
        n_p = sum(1 for a in acc if a["currently"] == "postdoc")
        sc  = sum(j["second_call_n"] for j in self.jobs)

        print(f"\n  -- Round {round_num} summary --")
        print(f"  accepted: {len(acc)}  (students={n_s}, postdocs={n_p})")
        print(f"  market remaining: {len(self.market)}")
        print(f"  open jobs: {sum(1 for j in self.jobs if not j['filled'])}")
        print(f"  total 2nd-calls so far: {sc}")
        if acc:
            mean_off = np.mean([len(a["all_offers"]) for a in acc])
            print(f"  mean cumulative offers (just-accepted): {mean_off:.2f}")

    # -------------------------------------------------------------------------
    # MARKET SUMMARY
    # -------------------------------------------------------------------------

    def _print_market_summary(self):
        if self.verbose < 1:
            return

        filled   = [a for a in self.applicants if a["accepted"]]
        n_s      = sum(1 for a in filled if a["currently"] == "student")
        n_p      = sum(1 for a in filled if a["currently"] == "postdoc")
        n_open   = sum(1 for j in self.jobs if not j["filled"])
        sc_total = sum(j["second_call_n"] for j in self.jobs)
        max_off  = max((len(a["all_offers"]) for a in self.applicants), default=0)

        print(f"\n{'=' * 60}")
        print(f"  MARKET CLOSED")
        print(f"{'=' * 60}")
        print(f"  Filled       : {len(filled)} / {self.n_positions}")
        print(f"  Students     : {n_s} / {self.n_student_applying}"
              f"  ({n_s / max(self.n_student_applying, 1) * 100:.1f}%)")
        print(f"  Postdocs     : {n_p} / {self.n_postdoc_reapplying}"
              f"  ({n_p / max(self.n_postdoc_reapplying, 1) * 100:.1f}%)")
        print(f"  Unfilled     : {n_open}")
        print(f"  2nd-calls    : {sc_total}")
        print(f"  Max offers   : {max_off}  (any single applicant)")
        print(f"{'=' * 60}")

    # -------------------------------------------------------------------------
    # RUN MARKET
    # -------------------------------------------------------------------------

    def run_market(self, offer_rounds: int = 10):
        """
        Full pipeline: assign counts -> build prestige -> build pools ->
        iterate offer rounds.  If inject_candidate() was called, prints
        injected candidate stats at the end.
        """
        if self.verbose >= 1:
            print(f"\n{'=' * 60}")
            print(f"  MARKET OPEN")
            print(f"{'=' * 60}")

        self.assign_application_counts()
        self.build_prestige()
        self.build_application_pools()

        for rnd in range(1, offer_rounds + 1):
            open_jobs = [j for j in self.jobs if not j["filled"]]
            if not open_jobs:
                if self.verbose >= 1:
                    print(f"\n  All jobs filled — stopping after round {rnd - 1}.")
                break
            if not self.market:
                if self.verbose >= 1:
                    print(f"\n  Market empty — stopping after round {rnd - 1}.")
                break

            if self.verbose >= 1:
                print(f"\n{'─' * 60}")
                print(f"  ROUND {rnd}  |  open={len(open_jobs)}  "
                      f"market={len(self.market)}")
                print(f"{'─' * 60}")

            self._offer_phase(rnd)
            accepted = self._accept_phase(rnd)
            self._print_round_summary(rnd, accepted)

        self._print_market_summary()

        if self._injected_idx is not None:
            self.print_injected_stats()

    # -------------------------------------------------------------------------
    # INJECTED CANDIDATE REPORTING
    # -------------------------------------------------------------------------

    def print_injected_stats(self):
        if self._injected_idx is None:
            print("No injected candidate.")
            return

        a         = self.applicants[self._injected_idx]
        n_total   = len(self.applicants)
        ovr_pct   = self._injected_idx / n_total * 100
        stu_list  = [x for x in self.applicants if x["currently"] == "student"]
        stu_rank  = sum(1 for x in self.applicants[:self._injected_idx]
                        if x["currently"] == "student")
        stu_pct   = stu_rank / max(len(stu_list), 1) * 100

        all_off  = a["all_offers"]
        n_sc_off = sum(1 for o in all_off if o["second_call"])
        n_orig   = a["n_applications"] - n_sc_off  # approximate original count

        print(f"\n{'=' * 55}")
        print(f"  INJECTED CANDIDATE OUTCOME")
        print(f"{'=' * 55}")
        print(f"  Student percentile : {stu_pct:.1f}%  "
              f"(rank {stu_rank} / {len(stu_list)})")
        print(f"  Overall percentile : {ovr_pct:.1f}%  "
              f"(rank {self._injected_idx} / {n_total})")
        print(f"  Skill              : {a['skill']:.4f}")
        print(f"  Final n_applications: {a['n_applications']}"
              f"  ({n_sc_off} added by 2nd-calls)")
        print(f"  Got >= 1 offer     : {'YES' if all_off else 'NO'}")
        print(f"  Total offers       : {len(all_off)}"
              f"  ({n_sc_off} from 2nd-call jobs)")

        if all_off:
            print(f"  Offers detail:")
            for o in all_off:
                sc = "  [2nd-call]" if o["second_call"] else ""
                print(f"    round {o['round']}: job {o['job_id']:3d}  "
                      f"prestige={o['prestige']:.3f}{sc}")

        if a["accepted"]:
            print(f"  Outcome: PLACED  "
                  f"job={a['accepted_job']}  "
                  f"prestige={a['accepted_prestige']:.3f}  "
                  f"round={a['accepted_round']}")
        else:
            print(f"  Outcome: NOT PLACED")
        print(f"{'=' * 55}")

    def get_injected_result(self) -> dict:
        """Injected candidate outcomes as a plain dict for aggregation."""
        if self._injected_idx is None:
            return {}
        a = self.applicants[self._injected_idx]
        return {
            "got_offer":            len(a["all_offers"]) > 0,
            "n_offers":             len(a["all_offers"]),
            "n_second_call_offers": sum(1 for o in a["all_offers"] if o["second_call"]),
            "accepted":             a["accepted"],
            "accepted_round":       a["accepted_round"],
            "accepted_prestige":    a["accepted_prestige"],
            "overall_percentile":   (self._injected_idx / len(self.applicants) * 100),
        }


# =============================================================================
# Multi-run helper
# =============================================================================

def run_injected_multi(
    student_percentile: float,
    n_applications:     int,
    n_runs:             int  = 100,
    offer_rounds:       int  = 10,
    cohort_kwargs:      dict = None,
    verbose_cohort:     int  = 0,
) -> dict:
    """Run N markets with an injected candidate; return aggregated stats."""
    if cohort_kwargs is None:
        cohort_kwargs = {}

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tqdm.auto import trange as _tr
        iterator = _tr(n_runs, desc=f"pct={student_percentile:.0f}% "
                                    f"n_apps={n_applications}")
    except ImportError:
        iterator = range(n_runs)

    results = []
    for _ in iterator:
        cohort = ApplicantCohort(**{**cohort_kwargs, "verbose": verbose_cohort})
        cohort.rank_applicants()
        cohort.inject_candidate(student_percentile, n_applications)
        cohort.run_market(offer_rounds=offer_rounds)
        results.append(cohort.get_injected_result())

    offer_rate  = np.mean([r["got_offer"] for r in results])
    accept_rate = np.mean([r["accepted"]  for r in results])
    mean_off    = np.mean([r["n_offers"]  for r in results])
    placed      = [r for r in results if r["accepted_prestige"] is not None]
    mean_pres   = np.mean([r["accepted_prestige"] for r in placed]) if placed else float("nan")
    mean_rnd    = np.mean([r["accepted_round"]    for r in placed]) if placed else float("nan")

    summary = dict(
        student_percentile=student_percentile,
        n_applications=n_applications,
        offer_rate=offer_rate,
        acceptance_rate=accept_rate,
        mean_n_offers=mean_off,
        mean_prestige=mean_pres,
        mean_round=mean_rnd,
        n_runs=n_runs,
    )

    print(f"\n{'=' * 50}")
    print(f"  pct={student_percentile:.0f}%  n_apps={n_applications}  "
          f"({n_runs} runs)")
    print(f"  Got >= 1 offer : {offer_rate * 100:.1f}%")
    print(f"  Accepted       : {accept_rate * 100:.1f}%")
    print(f"  Mean offers    : {mean_off:.2f}")
    print(f"  Mean prestige  : {mean_pres:.3f}")
    print(f"  Mean round     : {mean_rnd:.2f}")
    print(f"{'=' * 50}")
    return summary


# =============================================================================
# Grid sweep helpers
# =============================================================================

def _aggregate(raw_rows: List[dict]) -> List[dict]:
    """
    raw_rows: list of per-run dicts, each with keys
        student_percentile, n_applications, got_offer, accepted,
        accepted_prestige, accepted_round
    Returns one summary dict per (pct, n_apps) cell.
    """
    cells = defaultdict(list)
    for r in raw_rows:
        cells[(r["student_percentile"], r["n_applications"])].append(r)

    summaries = []
    for (pct, n_apps), rows in sorted(cells.items()):
        placed = [r for r in rows if r["accepted_prestige"] is not None]
        summaries.append({
            "student_percentile": pct,
            "n_applications":     n_apps,
            "offer_rate":         np.mean([r["got_offer"] for r in rows]),
            "acceptance_rate":    np.mean([r["accepted"]  for r in rows]),
            "mean_n_offers":      np.mean([r["n_offers"]  for r in rows]),
            "mean_prestige":      (np.mean([r["accepted_prestige"] for r in placed])
                                   if placed else float("nan")),
            "mean_round":         (np.mean([r["accepted_round"] for r in placed])
                                   if placed else float("nan")),
            "n_runs":             len(rows),
        })
    return summaries


def run_grid_sweep_serial(
    percentiles:   List[float] = [10, 25, 50, 75],
    n_apps_grid:   List[int]   = [5, 10, 15, 20, 30, 40, 60, 80],
    n_runs:        int         = 200,
    offer_rounds:  int         = 10,
    cohort_kwargs: dict        = None,
) -> List[dict]:
    """
    One injected candidate per market.
    Total markets = len(percentiles) * len(n_apps_grid) * n_runs.
    Cleanest isolation; slowest.
    """
    if cohort_kwargs is None:
        cohort_kwargs = {}

    n_cells = len(percentiles) * len(n_apps_grid)
    raw     = []

    for cell_i, pct in enumerate(percentiles):
        for n_apps in n_apps_grid:
            print(f"  [serial {cell_i * len(n_apps_grid) + n_apps_grid.index(n_apps) + 1}"
                  f"/{n_cells}]  pct={pct:.0f}%  n_apps={n_apps} ...")

            summary = run_injected_multi(
                student_percentile = pct,
                n_applications     = n_apps,
                n_runs             = n_runs,
                offer_rounds       = offer_rounds,
                cohort_kwargs      = cohort_kwargs,
                verbose_cohort     = 0,
            )
            raw.append(summary)

    return raw


def run_grid_sweep_per_n_apps(
    percentiles:   List[float] = [10, 25, 50, 75],
    n_apps_grid:   List[int]   = [5, 10, 15, 20, 30, 40, 60, 80],
    n_runs:        int         = 200,
    offer_rounds:  int         = 10,
    cohort_kwargs: dict        = None,
) -> List[dict]:
    """
    One market per n_apps value; all percentiles injected simultaneously.
    Total markets = len(n_apps_grid) * n_runs.
    """
    if cohort_kwargs is None:
        cohort_kwargs = {}

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tqdm.auto import trange as _tr
    except ImportError:
        _tr = None

    raw = []

    for n_apps in n_apps_grid:
        print(f"\n[per_n_apps batch]  n_apps={n_apps}  "
              f"({n_runs} runs, {len(percentiles)} percentiles each) ...")

        param_list = [(pct, n_apps) for pct in percentiles]
        iterator   = _tr(n_runs, desc=f"n_apps={n_apps}") if _tr else range(n_runs)

        for _ in iterator:
            cohort = ApplicantCohort(**{**cohort_kwargs, "verbose": 0})
            cohort.rank_applicants()
            cohort.inject_candidates_batch(param_list)
            cohort.run_market(offer_rounds=offer_rounds)
            raw.extend(cohort.get_all_injected_results())

    return _aggregate(raw)


def run_grid_sweep_full_batch(
    percentiles:   List[float] = [10, 25, 50, 75],
    n_apps_grid:   List[int]   = [5, 10, 15, 20, 30, 40, 60, 80],
    n_runs:        int         = 200,
    offer_rounds:  int         = 10,
    cohort_kwargs: dict        = None,
) -> List[dict]:
    """
    One market per run; all (pct, n_apps) cells injected simultaneously.
    Total markets = n_runs.  Fastest, but candidates interact.
    """
    if cohort_kwargs is None:
        cohort_kwargs = {}

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tqdm.auto import trange as _tr
    except ImportError:
        _tr = None

    param_list = [(pct, n_apps)
                  for pct in percentiles
                  for n_apps in n_apps_grid]

    print(f"\n[full batch]  {len(param_list)} cells  x  {n_runs} runs ...")
    iterator = _tr(n_runs, desc="full batch") if _tr else range(n_runs)

    raw = []
    for _ in iterator:
        cohort = ApplicantCohort(**{**cohort_kwargs, "verbose": 0})
        cohort.rank_applicants()
        cohort.inject_candidates_batch(param_list)
        cohort.run_market(offer_rounds=offer_rounds)
        raw.extend(cohort.get_all_injected_results())

    return _aggregate(raw)


# =============================================================================
# Threshold table building
# =============================================================================

def _isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators — returns non-decreasing fit."""
    y      = y.astype(float).copy()
    blocks = [[i, i, y[i]] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i + 1][2]:
            s, e = blocks[i][0], blocks[i + 1][1]
            blocks[i] = [s, e, float(y[s:e + 1].mean())]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    result = np.empty(len(y))
    for s, e, mn in blocks:
        result[s:e + 1] = mn
    return result


def _find_crossing(n_apps_grid: List[int],
                   rates: List[float],
                   target: float) -> Optional[int]:
    """Smallest n_apps where PAVA-smoothed rate >= target; None if never."""
    smoothed = _isotonic_increasing(np.array(rates))
    for n, r in zip(n_apps_grid, smoothed):
        if r >= target:
            return n
    return None


def build_threshold_table(
    results:      List[dict],
    percentiles:  List[float],
    n_apps_grid:  List[int],
    rate_key:     str   = "offer_rate",
    target_probs: Tuple = (50, 75, 90, 95, 99),
    max_apps:     int   = 200,
) -> dict:
    """Returns {target_prob -> {percentile -> n_apps_or_None}}"""
    lookup = {(r["student_percentile"], r["n_applications"]): r for r in results}
    table  = {}
    for prob in target_probs:
        table[prob] = {}
        for pct in percentiles:
            n_list = [n for n in n_apps_grid if (pct, n) in lookup]
            rates  = [lookup[(pct, n)][rate_key] for n in n_list]
            table[prob][pct] = _find_crossing(n_list, rates, prob / 100.0)
    return table


def _fmt(v: Optional[int], max_apps: int) -> str:
    return f">{max_apps}" if v is None else str(v)


def print_markdown_table(
    table:        dict,
    percentiles:  List[float],
    target_probs: Tuple,
    title:        str,
    delta_p:      float,
    max_apps:     int,
):
    """Rows = target prob, columns = student percentile."""
    col_hdrs  = [f"{int(p)}–{int(p + delta_p)}%" for p in percentiles]
    header    = "| Target prob | " + " | ".join(col_hdrs)  + " |"
    separator = "| --- | "        + " | ".join(["---"] * len(percentiles)) + " |"
    print(f"\n### {title}\n")
    print(header)
    print(separator)
    for prob in target_probs:
        cells = [_fmt(table[prob].get(pct), max_apps) for pct in percentiles]
        print(f"| P={prob}% | " + " | ".join(cells) + " |")


def print_tables(
    results:      List[dict],
    percentiles:  List[float] = None,
    n_apps_grid:  List[int]   = None,
    target_probs: Tuple       = (50, 75, 90, 95, 99),
    delta_p:      float       = 10.0,
    max_apps:     int         = 200,
):
    """Print both offer-rate and acceptance-rate threshold tables."""
    if percentiles is None:
        percentiles = sorted(set(r["student_percentile"] for r in results))
    if n_apps_grid is None:
        n_apps_grid = sorted(set(r["n_applications"] for r in results))

    for rate_key, label in [
        ("offer_rate",      "Applications for ≥1 offer  (empirical, injected candidate)"),
        ("acceptance_rate", "Applications to accept a job  (empirical, injected candidate)"),
    ]:
        tbl = build_threshold_table(results, percentiles, n_apps_grid,
                                    rate_key, target_probs, max_apps)
        print_markdown_table(tbl, percentiles, target_probs, label, delta_p, max_apps)


def print_raw_rates(
    results:     List[dict],
    percentiles: List[float] = None,
    n_apps_grid: List[int]   = None,
    rate_key:    str         = "offer_rate",
):
    """Raw empirical rates: rows = n_apps, columns = percentile."""
    if percentiles is None:
        percentiles = sorted(set(r["student_percentile"] for r in results))
    if n_apps_grid is None:
        n_apps_grid = sorted(set(r["n_applications"] for r in results))

    lookup    = {(r["student_percentile"], r["n_applications"]): r for r in results}
    col_hdrs  = [f"pct={int(p)}%" for p in percentiles]
    label     = "offer rate" if rate_key == "offer_rate" else "acceptance rate"

    print(f"\n### Raw {label} by (n_apps, percentile)\n")
    print("| n_apps | " + " | ".join(col_hdrs) + " |")
    print("| --- | "   + " | ".join(["---"] * len(percentiles)) + " |")
    for n in n_apps_grid:
        cells = [f"{lookup[(pct, n)][rate_key] * 100:.1f}%"
                 if (pct, n) in lookup else "—"
                 for pct in percentiles]
        print(f"| {n} | " + " | ".join(cells) + " |")


# =============================================================================
# Heatmap / scatter plotting
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

try:
    from tqdm import trange
except ImportError:
    trange = None


def collect_multi_run(
    cohort_kwargs:   dict,
    n_runs:          int  = 10,
    offer_rounds:    int  = 10,
    applicant_type:  str  = "both",
    verbose_cohort:  int  = 0,
    stochasticity:  float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run N independent markets and pool all applicants into flat arrays.

    Returns
    -------
    percentiles    : rank of each applicant expressed as a percentile (0–100)
    n_applications : applications sent by each applicant
    n_offers       : total offers received by each applicant
    """
    all_pct  = []
    all_apps = []
    all_off  = []

    iterator = trange(n_runs, desc="Running markets") if trange else range(n_runs)

    for run_i in iterator:
        if trange is None:
            print(f"  Market {run_i + 1}/{n_runs} ...", flush=True)

        kw = dict(cohort_kwargs)
        kw["verbose"] = verbose_cohort
        cohort = ApplicantCohort(**kw)
        cohort.rank_applicants()
        cohort.run_market(offer_rounds=offer_rounds)

        applicants = cohort.applicants
        n_total    = len(applicants)

        if applicant_type == "student":
            applicants = [a for a in applicants if a["currently"] == "student"]
        elif applicant_type == "postdoc":
            applicants = [a for a in applicants if a["currently"] == "postdoc"]

        ranks    = np.array([a["rank"]                  for a in applicants])
        n_apps   = np.array([a["n_applications"]        for a in applicants], dtype=float)
        n_offers = np.array([len(a["all_offers"]) for a in applicants], dtype=float)

        all_pct .append(ranks / n_total * 100)
        all_apps.append(n_apps)
        all_off .append(n_offers)

    return (
        np.concatenate(all_pct),
        np.concatenate(all_apps),
        np.concatenate(all_off),
    )


def plot_heatmap_from_arrays(
    percentiles:    np.ndarray,
    n_applications: np.ndarray,
    n_offers:       np.ndarray,
    n_rank_bins:    int            = 20,
    n_app_bins:     int            = 20,
    log_y:          bool           = True,
    ylim:           Optional[Tuple] = None,
    cmap:           str            = "plasma",
    min_count:      int            = 3,
    figsize:        Tuple          = (9, 6),
    title:          str            = "Mean offers by applicant rank & applications sent",
    save_path:      Optional[str]  = None,
):
    """Low-level heatmap function that works directly on numpy arrays."""
    x_edges = np.linspace(0, 100, n_rank_bins + 1)

    if log_y:
        if ylim is not None:
            y_min = np.log10(max(ylim[0], 1))
            y_max = np.log10(max(ylim[1], 1)) + 1e-9
        else:
            y_min = np.log10(max(n_applications.min(), 1))
            y_max = np.log10(n_applications.max()) + 1e-9
        y_edges = np.linspace(y_min, y_max, n_app_bins + 1)
        y_vals  = np.log10(np.clip(n_applications, 1, None))
    else:
        if ylim is not None:
            y_min, y_max = ylim
        else:
            y_min, y_max = n_applications.min(), n_applications.max()
        y_edges = np.linspace(y_min, y_max, n_app_bins + 1)
        y_vals  = n_applications.astype(float)

    x_idx = np.clip(np.digitize(percentiles, x_edges) - 1, 0, n_rank_bins - 1)
    y_idx = np.clip(np.digitize(y_vals,      y_edges) - 1, 0, n_app_bins  - 1)

    offer_sum = np.zeros((n_app_bins, n_rank_bins))
    counts    = np.zeros((n_app_bins, n_rank_bins))

    for xi, yi, o in zip(x_idx, y_idx, n_offers):
        offer_sum[yi, xi] += o
        counts[yi, xi]    += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_offers = np.where(counts >= min_count, offer_sum / counts, np.nan)

    masked = np.ma.masked_invalid(mean_offers)

    vmax = np.nanmax(mean_offers) if not np.all(np.isnan(mean_offers)) else 1.0
    vmin_pos = 1e-1

    n_colors  = 256
    base_cm   = plt.get_cmap(cmap)
    rgba_pos  = base_cm(np.linspace(0.0, 1.0, n_colors))
    white     = np.array([[1.0, 1.0, 1.0, 1.0]])
    all_rgba  = np.vstack([white, rgba_pos])
    cm        = mcolors.ListedColormap(all_rgba)
    cm.set_bad(color="#1a1a2e")

    pos_bounds = np.logspace(np.log10(vmin_pos), np.log10(vmax), n_colors + 1, base=10)
    boundaries = np.concatenate([[0.0], pos_bounds])
    norm       = mcolors.BoundaryNorm(boundaries, ncolors=n_colors + 1)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        x_edges, y_edges, masked,
        cmap=cm, shading="flat", norm=norm,
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean offers received", fontsize=11)
    int_ticks = np.arange(0, int(np.floor(vmax)) + 1)
    cbar.set_ticks(int_ticks)
    cbar.set_ticklabels([str(int(t)) for t in int_ticks])
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("Applicant percentile rank  (0 = best)", fontsize=12)

    if log_y:
        ax.set_ylabel("Applications sent  (log₁₀ scale)", fontsize=12)
        tick_vals = np.arange(np.floor(y_edges[0]), np.ceil(y_edges[-1]) + 1)
        ax.set_yticks(tick_vals)
        ax.set_yticklabels([f"{10**v:.0f}" for v in tick_vals], fontsize=9)
    else:
        ax.set_ylabel("Applications sent", fontsize=12)

    ax.set_xlim(0, 100)
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis="x", labelsize=9)

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_centres, y_centres)
    cs = ax.contour(Xc, Yc, counts, levels=[10, 30, 100],
                    colors="white", linewidths=0.6, alpha=0.45, linestyles="--")
    ax.clabel(cs, fmt="%d obs", fontsize=7, inline=True)

    ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig, ax


def plot_heatmap(
    cohort,
    applicant_type: str = "both",
    n_rank_bins: int = 20,
    n_app_bins: int = 20,
    log_y: bool = True,
    ylim: Optional[Tuple] = None,
    cmap: str = "plasma",
    min_count: int = 3,
    figsize: Tuple = (9, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Build a heatmap from a finished ApplicantCohort object."""
    applicants = cohort.applicants
    n_total    = len(applicants)

    if applicant_type == "student":
        applicants = [a for a in applicants if a["currently"] == "student"]
    elif applicant_type == "postdoc":
        applicants = [a for a in applicants if a["currently"] == "postdoc"]

    ranks    = np.array([a["rank"]                  for a in applicants])
    n_apps   = np.array([a["n_applications"]        for a in applicants], dtype=float)
    n_offers = np.array([len(a["all_offers"]) for a in applicants], dtype=float)

    percentiles = ranks / n_total * 100

    plot_heatmap_from_arrays(
        percentiles=percentiles,
        n_applications=n_apps,
        n_offers=n_offers,
        n_rank_bins=n_rank_bins,
        n_app_bins=n_app_bins,
        log_y=log_y,
        ylim=ylim,
        cmap=cmap,
        min_count=min_count,
        figsize=figsize,
        title=title or f"Offer rate by rank & applications sent  ({applicant_type})",
        save_path=save_path,
    )


def plot_heatmap_multi_run(
    cohort_kwargs:  dict  = None,
    n_runs:         int   = 10,
    offer_rounds:   int   = 10,
    applicant_type: str   = "both",
    n_rank_bins:    int   = 20,
    n_app_bins:     int   = 20,
    log_y:          bool  = True,
    ylim:           Optional[Tuple] = None,
    cmap:           str   = "plasma",
    min_count:      int   = 3,
    figsize:        Tuple = (9, 6),
    title:          Optional[str] = None,
    save_path:      Optional[str] = None,
    stochasticity:  float = 0.7,
):
    """Run N markets, pool results, and plot a heatmap."""
    if cohort_kwargs is None:
        cohort_kwargs = {}

    print(f"Simulating {n_runs} markets  (applicant_type='{applicant_type}') …")
    percentiles, n_apps, n_offers = collect_multi_run(
        cohort_kwargs=cohort_kwargs,
        n_runs=n_runs,
        offer_rounds=offer_rounds,
        applicant_type=applicant_type,
        stochasticity=stochasticity,
    )
    print(f"Pooled {len(percentiles):,} applicants across {n_runs} runs.")

    auto_title = (
        f"Mean offers by rank & applications sent"
        f"  ({applicant_type}, N={n_runs} markets)"
    )

    plot_heatmap_from_arrays(
        percentiles=percentiles,
        n_applications=n_apps,
        n_offers=n_offers,
        n_rank_bins=n_rank_bins,
        n_app_bins=n_app_bins,
        log_y=log_y,
        ylim=ylim,
        cmap=cmap,
        min_count=min_count,
        figsize=figsize,
        title=title or auto_title,
        save_path=save_path,
    )


def applications_needed_table(
    percentiles:    np.ndarray,
    n_applications: np.ndarray,
    n_offers:       np.ndarray,
    delta_p:        float             = 10.0,
    target_probs:   Tuple             = (50, 75, 90, 95, 99),
    max_apps:       int               = 500,
    print_table:    bool              = True,
) -> dict:
    """
    For each percentile band, estimate the number of applications needed to
    achieve each target probability of receiving at least one offer.
    """
    bands      = np.arange(0, 100, delta_p)
    results    = {}

    for band_start in bands:
        band_end = band_start + delta_p
        label    = f"{int(band_start)}–{int(band_end)}"

        mask = (percentiles >= band_start) & (percentiles < band_end)
        if mask.sum() == 0:
            results[label] = {p: None for p in target_probs}
            continue

        rates      = n_offers[mask] / n_applications[mask]
        p_per_app  = float(np.mean(rates))

        row = {}
        for prob in target_probs:
            p_target = prob / 100.0
            if p_per_app <= 0:
                row[prob] = None
            elif p_per_app >= 1:
                row[prob] = 1
            else:
                n = np.log(1 - p_target) / np.log(1 - p_per_app)
                row[prob] = int(np.ceil(n))
        results[label] = row

    if print_table:
        _print_apps_needed_markdown(results, target_probs, delta_p, max_apps)

    return results


def _print_apps_needed_markdown(
    results:      dict,
    target_probs: Tuple,
    delta_p:      float,
    max_apps:     int,
):
    """Print results as a GitHub-flavoured markdown table."""
    prob_headers = [f"P={p}%" for p in target_probs]
    header       = "| Percentile band | " + " | ".join(prob_headers) + " |"
    separator    = "| --- | " + " | ".join(["---"] * len(target_probs)) + " |"

    print(f"\n### Applications needed for ≥1 offer  (band width = {int(delta_p)}%)\n")
    print(header)
    print(separator)

    for label, row in results.items():
        cells = []
        for p in target_probs:
            v = row[p]
            if v is None:
                cells.append("N/A")
            elif v > max_apps:
                cells.append(f">{max_apps}")
            else:
                cells.append(str(v))
        print(f"| {label} | " + " | ".join(cells) + " |")


def applications_needed_table_multi_run(
    cohort_kwargs:  dict              = None,
    n_runs:         int               = 10,
    offer_rounds:   int               = 10,
    applicant_type: str               = "both",
    delta_p:        float             = 10.0,
    target_probs:   Tuple             = (50, 75, 90, 95, 99),
    max_apps:       int               = 200,
    stochasticity:  float = 0.7,
) -> dict:
    """Run N markets, pool applicants, then print the applications-needed table."""
    if cohort_kwargs is None:
        cohort_kwargs = {}

    print(f"Simulating {n_runs} markets  (applicant_type='{applicant_type}') …")
    pct, apps, offers = collect_multi_run(
        cohort_kwargs=cohort_kwargs,
        n_runs=n_runs,
        offer_rounds=offer_rounds,
        applicant_type=applicant_type,
        stochasticity=stochasticity,
    )
    print(f"Pooled {len(pct):,} applicants across {n_runs} runs.")

    return applications_needed_table(
        percentiles=pct,
        n_applications=apps,
        n_offers=offers,
        delta_p=delta_p,
        target_probs=target_probs,
        max_apps=max_apps,
        print_table=True,
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    cohort = ApplicantCohort(verbose=2)
    cohort.rank_applicants()
    cohort.inject_candidate(student_percentile=10, n_applications=15)
    cohort.run_market(offer_rounds=10)
