"""
TREND TO TILL, STEP 2b: product-success reinforcement learning pipeline
================================================

Framing (contextual bandit, trained with policy gradient / REINFORCE)
---------------------------------------------------------------------
* State / context : a product's attribute scores + the market (a UK region, or "UK" overall)
* Action          : back the product (predict SUCCESS = 1) or pass (predict FAIL = 0)
* Reward          : arrives later from the sales source, once the product has `horizon_months`
                    of sales in that market:
                        correct call            -> +1
                        backed a flop (FP)      -> -fp_cost
                        passed on a hit (FN)    -> -fn_cost
* Policy          : pi(back | x, r) = sigmoid( (w + d_r) . x + b_r )
                        w   = global attribute weights   (initialised from attributes.csv)
                        d_r = region-specific adjustments (kept small by L2)
                        b_r = region bias
* Learning        : REINFORCE with a per-region running-reward baseline.
                    Phase 1 (offline): replay historical outcomes in launch order for several epochs.
                    Phase 2 (online) : as new monthly sales arrive, any product/region that reaches the
                                       horizon becomes a new episode and the weights are updated.

Success definition
------------------
A product "succeeds" in a market if units sold in its first `horizon_months` months after launch
are at or above that market's threshold. Thresholds are the `success_percentile` of historical
first-N-month sales per market, fixed at training time (stored in the model).

Usage
-----
    python step2b_predict_success.py train   --data data --model model         # offline learning (+ holdout report)
    python step2b_predict_success.py update  --data data --model model --incoming data/incoming_sales.csv
    python step2b_predict_success.py predict --data data --model model --products data/candidate_products.csv
    python step2b_predict_success.py demo    --data data --model model         # all three in sequence
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

UK = "UK"


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
@dataclass
class Config:
    horizon_months: int = 6          # months of sales used to judge success
    success_percentile: float = 50   # market threshold = this percentile of historical sales
    prior_scale: float = 4.0         # converts attributes.csv weights (sum ~1) into starting logits
    lr_offline: float = 0.05
    lr_online: float = 0.1          # higher than the end of offline training so it can track drift
    epochs: int = 60
    fp_cost: float = 1.0             # penalty for backing a product that flops
    fn_cost: float = 1.0             # penalty for passing on a product that succeeds
    l2_global: float = 5e-3          # shrinks weights toward 0 (stops them growing without bound)
    l2_region: float = 5e-2          # keeps regional adjustments close to the global weights
    reward_mode: str = "magnitude"   # "magnitude": reward scaled by how big the hit/flop was
                                     #   (|log(units / threshold)|, clipped); "binary": +/-1 only
    gradient: str = "expected"       # "expected": all-action policy gradient (low variance; valid because
                                     #   the sales outcome tells us the reward of BOTH actions)
                                     # "sampled" : classic REINFORCE on the sampled action only
    baseline_momentum: float = 0.05
    decision_threshold: float = 0.5
    seed: int = 7


# ----------------------------------------------------------------------------------------------
# Data loading and outcome (reward) construction
# ----------------------------------------------------------------------------------------------
def load_attributes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"attribute", "initial_weight"}
    if missing := required - set(df.columns):
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def load_products(path: Path, attributes: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if missing := set(attributes) - set(df.columns):
        raise ValueError(f"{path} is missing attribute columns: {missing}")
    df["launch_period"] = pd.PeriodIndex.from_fields(year=df.launch_year, month=df.launch_month, freq="M")
    return df.set_index("product_id")


def load_sales(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"product_id", "region", "year", "month", "units_sold"}
    if missing := required - set(df.columns):
        raise ValueError(f"{path} is missing columns: {missing}")
    df["period"] = pd.PeriodIndex.from_fields(year=df.year, month=df.month, freq="M")
    return df


def with_uk_total(sales: pd.DataFrame) -> pd.DataFrame:
    """Append a national 'UK' row per product-month (sum over regions)."""
    uk = sales.groupby(["product_id", "period"], as_index=False)["units_sold"].sum()
    uk["region"] = UK
    return pd.concat([sales[["product_id", "region", "period", "units_sold"]], uk], ignore_index=True)


def build_outcomes(sales: pd.DataFrame, products: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Units sold in the first `horizon` months per product per market, and whether the window is complete."""
    s = with_uk_total(sales).merge(products[["launch_period"]], left_on="product_id", right_index=True)
    s["age"] = (s.period - s.launch_period).apply(lambda d: d.n)
    s = s[(s.age >= 0) & (s.age < horizon)]
    out = s.groupby(["product_id", "region"]).agg(
        units=("units_sold", "sum"), months_seen=("age", "nunique"), launch_period=("launch_period", "first")
    ).reset_index()
    out["complete"] = out.months_seen >= horizon
    out["outcome_period"] = out.launch_period + (horizon - 1)   # when the reward becomes known
    return out


def features(products: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    """Scores 0-10 -> centred, roughly in [-2, 2]."""
    return (products[attributes].astype(float) - 5.0) / 2.5


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ----------------------------------------------------------------------------------------------
# The agent
# ----------------------------------------------------------------------------------------------
@dataclass
class PolicyGradientAgent:
    attributes: list[str]
    markets: list[str]
    config: Config
    w: np.ndarray = None                 # global attribute weights
    d: np.ndarray = None                 # region adjustments  (markets x attributes)
    b: np.ndarray = None                 # region bias
    baseline: np.ndarray = None          # running mean reward per market
    cal: np.ndarray = None               # Platt calibration [slope, intercept] for reported probabilities
    episodes_seen: int = 0
    thresholds: dict = field(default_factory=dict)
    rng: np.random.Generator = None

    @classmethod
    def from_priors(cls, attr_df: pd.DataFrame, markets: list[str], config: Config):
        k, m = len(attr_df), len(markets)
        prior = attr_df.initial_weight.to_numpy(float)
        prior = prior / prior.sum() * config.prior_scale
        return cls(attributes=list(attr_df.attribute), markets=list(markets), config=config,
                   w=prior.copy(), d=np.zeros((m, k)), b=np.zeros(m), baseline=np.zeros(m),
                   cal=np.array([1.0, 0.0]), rng=np.random.default_rng(config.seed))

    # --- policy -----------------------------------------------------------
    def _idx(self, market: str) -> int:
        if market not in self.markets:            # unseen region: start it from the global weights
            self.markets.append(market)
            self.d = np.vstack([self.d, np.zeros(len(self.attributes))])
            self.b = np.append(self.b, 0.0)
            self.baseline = np.append(self.baseline, 0.0)
        return self.markets.index(market)

    def logit(self, x: np.ndarray, market: str) -> float:
        i = self._idx(market)
        return float(x @ (self.w + self.d[i]) + self.b[i])

    def policy_prob(self, x: np.ndarray, market: str) -> float:
        """Probability the policy backs the product (used for acting and learning)."""
        return _sigmoid(self.logit(x, market))

    def prob(self, x: np.ndarray, market: str) -> float:
        """Calibrated probability of success (used for reporting and predictions)."""
        return _sigmoid(self.cal[0] * self.logit(x, market) + self.cal[1])

    def act(self, x: np.ndarray, market: str) -> tuple[int, float]:
        p = self.policy_prob(x, market)
        return int(self.rng.random() < p), p     # stochastic policy = built-in exploration

    def reward(self, action: int, success: int, size: float = 1.0) -> float:
        """size = how decisive the outcome was (1.0 for binary rewards)."""
        if action == success:
            return size
        return -size * (self.config.fp_cost if action == 1 else self.config.fn_cost)

    def outcome_size(self, units: float, threshold: float) -> float:
        if self.config.reward_mode != "magnitude":
            return 1.0
        return float(np.clip(abs(np.log((units + 1) / (threshold + 1))), 0.25, 2.0))

    # --- learning ---------------------------------------------------------
    def learn(self, x: np.ndarray, market: str, success: int, lr: float, size: float = 1.0) -> dict:
        """One policy-gradient step: act, receive reward, push log-prob of the action by its advantage."""
        i = self._idx(market)
        cfg = self.config
        action, p = self.act(x, market)
        r = self.reward(action, success, size)
        if cfg.gradient == "sampled":
            advantage = r - self.baseline[i]
            grad_logit = advantage * (action - p)      # d log pi(a) / d logit for a Bernoulli policy
        else:
            # E_a[(r(a) - b) * dlog pi(a)/dlogit] = p(1-p) * (r(back) - r(pass)); baseline cancels
            grad_logit = p * (1 - p) * (self.reward(1, success, size) - self.reward(0, success, size))
        self.w += lr * (grad_logit * x - cfg.l2_global * self.w)
        if market != UK:
            self.d[i] += lr * (grad_logit * x - cfg.l2_region * self.d[i])
        self.b[i] += lr * grad_logit
        self.baseline[i] += cfg.baseline_momentum * (r - self.baseline[i])
        self.episodes_seen += 1
        return {"action": action, "p": p, "reward": r}

    def calibrate(self, xs: list[np.ndarray], markets: list[str], ys: np.ndarray, steps: int = 500):
        """Fit Platt scaling so reported probabilities match observed success rates."""
        z = np.array([self.logit(x, m) for x, m in zip(xs, markets)])
        a, c = 1.0, 0.0
        for _ in range(steps):
            err = _sigmoid(a * z + c) - ys
            a -= 0.5 * float((err * z).mean())
            c -= 0.5 * float(err.mean())
        self.cal = np.array([a, c])

    def calibrate_step(self, x: np.ndarray, market: str, success: int, lr: float = 0.01):
        z = self.logit(x, market)
        err = _sigmoid(self.cal[0] * z + self.cal[1]) - success
        self.cal -= lr * err * np.array([z, 1.0])

    # --- inspection -------------------------------------------------------
    def weight_table(self, prior: pd.Series | None = None) -> pd.DataFrame:
        imp = np.abs(self.w) / np.abs(self.w).sum()
        df = pd.DataFrame({"attribute": self.attributes, "learned_coefficient": self.w.round(4),
                           "learned_weight": imp.round(4),
                           "direction": np.where(self.w >= 0, "helps", "hurts")})
        if prior is not None:
            df.insert(1, "initial_weight", prior.to_numpy())
        for i, m in enumerate(self.markets):
            if m != UK:
                df[f"coef_{m}"] = (self.w + self.d[i]).round(4)
        return df

    def explain(self, x: np.ndarray, market: str, top: int = 3) -> str:
        i = self._idx(market)
        contrib = x * (self.w + self.d[i])
        order = np.argsort(-np.abs(contrib))[:top]
        return "; ".join(f"{self.attributes[j]} {'+' if contrib[j] >= 0 else '-'}{abs(contrib[j]):.2f}" for j in order)

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {"attributes": self.attributes, "markets": self.markets, "config": asdict(self.config),
                "w": self.w.tolist(), "d": self.d.tolist(), "b": self.b.tolist(),
                "baseline": self.baseline.tolist(), "cal": self.cal.tolist(), "episodes_seen": self.episodes_seen,
                "thresholds": self.thresholds}

    @classmethod
    def from_dict(cls, s: dict):
        cfg = Config(**s["config"])
        return cls(attributes=s["attributes"], markets=s["markets"], config=cfg, w=np.array(s["w"]),
                   d=np.array(s["d"]), b=np.array(s["b"]), baseline=np.array(s["baseline"]),
                   cal=np.array(s.get("cal", [1.0, 0.0])),
                   episodes_seen=s["episodes_seen"], thresholds=s["thresholds"],
                   rng=np.random.default_rng(cfg.seed + s["episodes_seen"]))


# ----------------------------------------------------------------------------------------------
# Training, evaluation, online updates, prediction
# ----------------------------------------------------------------------------------------------
def label(outcomes: pd.DataFrame, thresholds: dict) -> pd.Series:
    return (outcomes.units >= outcomes.region.map(thresholds)).astype(int)


def compute_thresholds(outcomes: pd.DataFrame, pct: float) -> dict:
    done = outcomes[outcomes.complete]
    return done.groupby("region").units.quantile(pct / 100).round(1).to_dict()


def train_offline(agent: PolicyGradientAgent, episodes: pd.DataFrame, X: pd.DataFrame, log_every: int = 10):
    """Replay historical episodes (in the order their rewards became known) for several epochs."""
    cfg = agent.config
    episodes = episodes.sort_values("outcome_period")
    for epoch in range(cfg.epochs):
        lr = cfg.lr_offline * (1 - 0.8 * epoch / max(cfg.epochs - 1, 1))   # linear decay to 20%
        rewards = [agent.learn(X.loc[e.product_id].to_numpy(), e.region, e.success, lr,
                               agent.outcome_size(e.units, agent.thresholds[e.region]))["reward"]
                   for e in episodes.itertuples()]
        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch + 1:3d}  mean reward {np.mean(rewards):+.3f}  lr {lr:.4f}")
    agent.calibrate([X.loc[p].to_numpy() for p in episodes.product_id], list(episodes.region),
                    episodes.success.to_numpy())


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def evaluate(agent: PolicyGradientAgent, episodes: pd.DataFrame, X: pd.DataFrame) -> dict:
    p = np.array([agent.prob(X.loc[e.product_id].to_numpy(), e.region) for e in episodes.itertuples()])
    y = episodes.success.to_numpy()
    pred = (p >= agent.config.decision_threshold).astype(int)
    tp, fp = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    return {"n": len(y), "accuracy": round(float((pred == y).mean()), 3), "auc": round(auc(y, p), 3),
            "brier": round(float(((p - y) ** 2).mean()), 3),
            "precision": round(float(tp / max(tp + fp, 1)), 3), "recall": round(float(tp / max(tp + fn, 1)), 3)}


def cmd_train(args, cfg: Config):
    data, model_dir = Path(args.data), Path(args.model)
    model_dir.mkdir(parents=True, exist_ok=True)
    attr_df = load_attributes(data / "attributes.csv")
    attrs = list(attr_df.attribute)
    products = load_products(data / "products.csv", attrs)
    sales = load_sales(data / "historical_sales.csv")
    X = features(products, attrs)

    outcomes = build_outcomes(sales, products, cfg.horizon_months)
    markets = [UK] + sorted(sales.region.unique())

    # 1) honest holdout: learn thresholds + weights on earlier launches, test on later ones
    if args.holdout_from:
        cut = pd.Period(args.holdout_from, freq="M")
        done = outcomes[outcomes.complete]
        train_ep, test_ep = done[done.launch_period < cut].copy(), done[done.launch_period >= cut].copy()
        th = compute_thresholds(train_ep, cfg.success_percentile)
        train_ep["success"], test_ep["success"] = label(train_ep, th), label(test_ep, th)
        prior_agent = PolicyGradientAgent.from_priors(attr_df, markets, cfg)
        trial = PolicyGradientAgent.from_priors(attr_df, markets, cfg)
        trial.thresholds = th
        print(f"Holdout check: train on launches before {cut} ({len(train_ep)} episodes), "
              f"test on {len(test_ep)} later episodes")
        train_offline(trial, train_ep, X, log_every=0)
        m_prior, m_rl = evaluate(prior_agent, test_ep, X), evaluate(trial, test_ep, X)
        print("  prior weights only :", m_prior)
        print("  after RL training  :", m_rl)
        _record_metrics(model_dir, "holdout_prior_weights_only", f"launches from {cut}", m_prior, reset=True)
        _record_metrics(model_dir, "holdout_after_rl_training", f"launches from {cut}", m_rl)

    # 2) final model on all complete history
    thresholds = compute_thresholds(outcomes, cfg.success_percentile)
    episodes = outcomes[outcomes.complete].copy()
    episodes["success"] = label(episodes, thresholds)
    agent = PolicyGradientAgent.from_priors(attr_df, markets, cfg)
    agent.thresholds = thresholds
    print(f"\nOffline training on {len(episodes)} historical episodes "
          f"({episodes.product_id.nunique()} products x {len(markets)} markets)")
    train_offline(agent, episodes, X)
    m_in = evaluate(agent, episodes, X)
    print("  in-sample:", m_in)
    _record_metrics(model_dir, "offline_in_sample", "all historical episodes", m_in, reset=not args.holdout_from)

    # outcomes still in progress at the end of history -> pending buffer for online learning
    pending = outcomes[~outcomes.complete]
    sales_tail = with_uk_total(sales)
    sales_tail = sales_tail[sales_tail.product_id.isin(pending.product_id)]
    pending_state = _pending_from_sales(sales_tail, products, cfg.horizon_months)

    out_hist = episodes.assign(threshold=episodes.region.map(thresholds))
    out_hist[["product_id", "region", "launch_period", "units", "threshold", "success"]].to_csv(
        model_dir / "historical_outcomes.csv", index=False)
    save_state(model_dir, agent, pending_state, done_keys=set())
    wt = agent.weight_table(attr_df.set_index("attribute").initial_weight)
    wt.to_csv(model_dir / "learned_weights.csv", index=False)
    _append_history(model_dir, agent, stage="offline", period=str(episodes.outcome_period.max()), reset=True)
    print("\nLearned global weights:")
    print(wt[["attribute", "initial_weight", "learned_weight", "learned_coefficient", "direction"]]
          .sort_values("learned_weight", ascending=False).to_string(index=False))


def _pending_from_sales(sales_uk: pd.DataFrame, products: pd.DataFrame, horizon: int) -> dict:
    s = sales_uk.merge(products[["launch_period"]], left_on="product_id", right_index=True)
    s["age"] = (s.period - s.launch_period).apply(lambda d: d.n)
    s = s[(s.age >= 0) & (s.age < horizon)]
    pending = {}
    for (pid, region), g in s.groupby(["product_id", "region"]):
        pending[f"{pid}|{region}"] = {"units": float(g.units_sold.sum()), "ages": sorted(set(map(int, g.age)))}
    return pending


def cmd_update(args, cfg_unused: Config):
    """Consume newly arrived sales month by month; learn from every outcome that completes."""
    data, model_dir = Path(args.data), Path(args.model)
    agent, pending, done_keys = load_state(model_dir)
    cfg = agent.config
    products = load_products(Path(args.products) if args.products else data / "products.csv", agent.attributes)
    X = features(products, agent.attributes)
    new = with_uk_total(load_sales(Path(args.incoming)))
    new = new.merge(products[["launch_period"]], left_on="product_id", right_index=True, how="inner")
    new["age"] = (new.period - new.launch_period).apply(lambda d: d.n)

    log = []
    for period, batch in new.sort_values("period").groupby("period"):
        batch = batch[(batch.age >= 0) & (batch.age < cfg.horizon_months)]
        completed = []
        for e in batch.itertuples():
            key = f"{e.product_id}|{e.region}"
            if key in done_keys:
                continue
            slot = pending.setdefault(key, {"units": 0.0, "ages": []})
            if e.age not in slot["ages"]:            # ignore duplicate deliveries of the same month
                slot["units"] += float(e.units_sold)
                slot["ages"].append(int(e.age))
            if len(slot["ages"]) >= cfg.horizon_months:
                completed.append(key)
        for key in completed:
            pid, region = key.split("|")
            units = pending.pop(key)["units"]
            done_keys.add(key)
            threshold = agent.thresholds.get(region)
            if threshold is None:                    # brand-new region: no threshold yet, skip reward
                continue
            success = int(units >= threshold)
            x = X.loc[pid].to_numpy()
            p_before = agent.prob(x, region)         # prediction made before seeing the outcome
            agent.learn(x, region, success, cfg.lr_online, agent.outcome_size(units, threshold))
            agent.calibrate_step(x, region, success)
            log.append({"period": str(period), "product_id": pid, "region": region, "units": units,
                        "threshold": threshold, "success": success, "p_before": round(p_before, 3),
                        "correct_before": int((p_before >= cfg.decision_threshold) == success)})
        n_new = len(completed)
        if n_new:
            recent = [r["correct_before"] for r in log if r["period"] == str(period)]
            print(f"  {period}: {n_new:3d} outcomes learned  |  accuracy of predictions before update "
                  f"{np.mean(recent):.2f}")
            _append_history(model_dir, agent, stage="online", period=str(period))
        else:
            print(f"  {period}: sales received, no product reached its {cfg.horizon_months}-month mark yet")

    save_state(model_dir, agent, pending, done_keys)
    if log:
        log_df = pd.DataFrame(log)
        path = model_dir / "online_learning_log.csv"
        log_df.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"\nOnline: {len(log_df)} new outcomes; pre-update accuracy {log_df.correct_before.mean():.3f}")
        y, p = log_df.success.to_numpy(), log_df.p_before.to_numpy()
        _record_metrics(model_dir, "online_before_update", f"{log_df.period.min()} to {log_df.period.max()}",
                        {"n": len(y), "accuracy": round(float(log_df.correct_before.mean()), 3),
                         "auc": round(auc(y, p), 3), "brier": round(float(((p - y) ** 2).mean()), 3)})
    attr_prior = load_attributes(data / "attributes.csv").set_index("attribute").initial_weight
    agent.weight_table(attr_prior).to_csv(model_dir / "learned_weights.csv", index=False)


def cmd_predict(args, cfg_unused: Config):
    model_dir = Path(args.model)
    agent, _, _ = load_state(model_dir)
    cfg = agent.config
    products = load_products(Path(args.products), agent.attributes)
    X = features(products, agent.attributes)
    regions = [m for m in agent.markets if m != UK]
    rows = []
    for pid, prod in products.iterrows():
        x = X.loc[pid].to_numpy()
        p_uk = agent.prob(x, UK)
        reg = {r: agent.prob(x, r) for r in regions}
        likely = [r for r, p in sorted(reg.items(), key=lambda kv: -kv[1]) if p >= cfg.decision_threshold]
        rows.append({"product_id": pid, "product_name": prod.get("product_name", ""),
                     "uk_success_probability": round(p_uk, 3), "uk_verdict": verdict(p_uk),
                     "regions_likely_to_succeed": ", ".join(likely) or "none",
                     "top_drivers_uk": agent.explain(x, UK),
                     **{f"p_{r}": round(p, 3) for r, p in reg.items()}})
    out = pd.DataFrame(rows).sort_values("uk_success_probability", ascending=False)
    path = model_dir / "predictions.csv"
    out.to_csv(path, index=False)
    print(out[["product_id", "product_name", "uk_success_probability", "uk_verdict",
               "regions_likely_to_succeed"]].to_string(index=False))
    print(f"\nFull regional breakdown written to {path}")


def verdict(p: float) -> str:
    return "Likely success" if p >= 0.65 else "Likely fail" if p < 0.35 else "Uncertain - test further"


# ----------------------------------------------------------------------------------------------
# State persistence
# ----------------------------------------------------------------------------------------------
def save_state(model_dir: Path, agent: PolicyGradientAgent, pending: dict, done_keys: set):
    state = {"agent": agent.to_dict(), "pending": pending, "done_keys": sorted(done_keys)}
    (model_dir / "model_state.json").write_text(json.dumps(state, indent=1))


def load_state(model_dir: Path):
    path = model_dir / "model_state.json"
    if not path.exists():
        raise SystemExit(f"No model found at {path}. Run `train` first.")
    s = json.loads(path.read_text())
    return PolicyGradientAgent.from_dict(s["agent"]), s["pending"], set(s["done_keys"])


def _record_metrics(model_dir: Path, stage: str, scope: str, metrics: dict, reset: bool = False):
    """Append evaluation results to model/evaluation_metrics.csv."""
    path = model_dir / "evaluation_metrics.csv"
    row = pd.DataFrame([{"stage": stage, "evaluated_on": scope, **metrics}])
    row.to_csv(path, mode="w" if reset else "a", header=reset or not path.exists(), index=False)


def _append_history(model_dir: Path, agent: PolicyGradientAgent, stage: str, period: str, reset: bool = False):
    imp = np.abs(agent.w) / np.abs(agent.w).sum()
    row = {"stage": stage, "period": period, "episodes_seen": agent.episodes_seen,
           **{a: round(float(v), 4) for a, v in zip(agent.attributes, imp)}}
    path = model_dir / "weights_history.csv"
    pd.DataFrame([row]).to_csv(path, mode="w" if reset else "a", header=reset or not path.exists(), index=False)


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "update", "predict", "demo"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="model")
    ap.add_argument("--incoming", default=None, help="CSV of newly arrived sales (update)")
    ap.add_argument("--products", default=None, help="products CSV (predict: products to score)")
    ap.add_argument("--holdout-from", default="2024-07", help="YYYY-MM; set to '' to skip the holdout check")
    ap.add_argument("--horizon", type=int, default=Config.horizon_months)
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    args = ap.parse_args()
    cfg = Config(horizon_months=args.horizon, epochs=args.epochs)

    if args.command == "train":
        cmd_train(args, cfg)
    elif args.command == "update":
        args.incoming = args.incoming or str(Path(args.data) / "incoming_sales.csv")
        cmd_update(args, cfg)
    elif args.command == "predict":
        args.products = args.products or str(Path(args.data) / "candidate_products.csv")
        cmd_predict(args, cfg)
    else:
        print("=== 1. Offline training on historical data ===")
        cmd_train(args, cfg)
        print("\n=== 2. Online learning from incoming sales ===")
        args.incoming = str(Path(args.data) / "incoming_sales.csv")
        args.products = None
        cmd_update(args, cfg)
        print("\n=== 3. Predictions for candidate products ===")
        args.products = str(Path(args.data) / "candidate_products.csv")
        cmd_predict(args, cfg)


if __name__ == "__main__":
    main()
