"""fl_core.py — leak-free federated training harness (FedAvg, FedProx, SCAFFOLD, FedDyn, FedAdapt-F).
Every run: fresh model from its seed; local SGD+momentum reset for every client (no carried optimiser state); partitions on FLOWS."""
import json, os, time, hashlib, gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import f1_score

N_CLASSES = 8


def build_model(n_feat, attention=True):
    inp = layers.Input(shape=(n_feat,), name='x'); x = inp
    if attention:
        a = layers.Softmax(name='feat_attention')(layers.Dense(n_feat, name='feat_attn_score')(inp))
        x = layers.Lambda(lambda t: t[0] * t[1] * n_feat, name='attn_reweight')([inp, a])
    h = layers.Dense(256, activation='relu')(x); h = layers.Dropout(0.2)(h)
    h = layers.Dense(128, activation='relu')(h)
    return Model(inp, layers.Dense(N_CLASSES, activation='softmax')(h))


def wsig(ws):
    return hashlib.md5(b''.join(np.ascontiguousarray(w).tobytes() for w in ws)).hexdigest()[:12]


# ------------------------------- partitions (on flow indices of the training split) -------------------------------
def partition(y, kind, K, seed, alpha=None, groups=None, min_size=50):
    rng = np.random.default_rng(seed); n = len(y)
    if kind == 'iid':
        p = rng.permutation(n); return [np.sort(c) for c in np.array_split(p, K)]
    if kind == 'dirichlet':
        for _ in range(100):
            parts = [[] for _ in range(K)]
            for c in np.unique(y):
                idx = np.where(y == c)[0]; rng.shuffle(idx)
                cuts = (np.cumsum(rng.dirichlet(alpha * np.ones(K))) * len(idx)).astype(int)[:-1]
                for k, s in enumerate(np.split(idx, cuts)): parts[k].extend(s.tolist())
            if min(len(p) for p in parts) >= min_size: return [np.sort(np.array(p)) for p in parts]
        raise RuntimeError('could not draw a Dirichlet partition with min client size')
    if kind == 'ip':   # natural clients: whole source-IP groups, greedy size balancing, seed-shuffled order
        ug, cnt = np.unique(groups, return_counts=True)
        order = rng.permutation(len(ug)); order = order[np.argsort(-cnt[order], kind='stable')]
        load = np.zeros(K, dtype=int); owner = {}
        for i in order:
            k = int(np.argmin(load)); owner[ug[i]] = k; load[k] += cnt[i]
        own = np.array([owner[g] for g in groups])
        return [np.where(own == k)[0] for k in range(K)]
    raise ValueError(kind)


def partition_sig(parts):
    return hashlib.md5(b''.join(np.asarray(p, dtype=np.int64).tobytes() for p in parts)).hexdigest()[:12]


# ------------------------------- federated run -------------------------------
def run_fl(X, y, Xv, yv, Xt, yt, parts, method, seed, rounds=50, local_epochs=1, batch=128, lr=0.03,
           momentum=0.9, mu=0.01, dyn_alpha=0.01, attention=True, eval_every=5, log=None, clip_norm=1.0):
    """method in {fedavg, fedprox, scaffold, feddyn, fedadapt, fedadapt_pure}.
    Local optimiser: SGD with momentum; the momentum buffer is reset to zero at the start of EVERY client's
    local training, so no optimiser state is carried between clients or rounds.
    clip_norm: global-norm gradient clipping applied to every local step (None = off); needed because the
    standardised CIC features are heavy-tailed (values of hundreds of standard deviations)."""
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    model = build_model(X.shape[1], attention)
    tv = model.trainable_variables
    assert len(model.weights) == len(tv), 'model has non-trainable state; aggregation assumes none'
    att_model = Model(model.input, model.get_layer('feat_attention').output) if attention else None
    K = len(parts); n_k = np.array([len(p) for p in parts], dtype=np.float64)
    _pk = [np.bincount(y[p], minlength=N_CLASSES) / max(len(p), 1) for p in parts]
    lab_ent = np.array([float(-(q[q > 0] * np.log(q[q > 0])).sum()) for q in _pk])   # client label-distribution entropy
    gw = [w.numpy().copy() for w in tv]; init_sig = wsig(gw)
    zeros = lambda: [np.zeros_like(w) for w in gw]
    ref = [tf.Variable(w, trainable=False) for w in gw]          # global weights at round start
    corr = [tf.Variable(np.zeros_like(w), trainable=False) for w in gw]   # scaffold c - c_i  /  feddyn h_k
    c_glob, c_loc = zeros(), [zeros() for _ in range(K)]
    dyn_h_loc, dyn_h_srv = [zeros() for _ in range(K)], zeros()
    buf = [tf.Variable(np.zeros_like(w), trainable=False) for w in gw]   # local momentum buffer
    gsum = [tf.Variable(np.zeros_like(w), trainable=False) for w in gw]  # sum of raw local gradients (SCAFFOLD)
    lossf = tf.keras.losses.SparseCategoricalCrossentropy()
    lr_t, mu_t, da_t = tf.constant(lr, tf.float32), tf.constant(mu, tf.float32), tf.constant(dyn_alpha, tf.float32)
    mom_t = tf.constant(momentum, tf.float32)

    @tf.function(reduce_retracing=True)
    def local_epoch(Xc, yc, perm):
        nb = tf.shape(perm)[0] // batch
        for i in tf.range(nb):
            bi = perm[i * batch:(i + 1) * batch]
            xb, yb = tf.gather(Xc, bi), tf.gather(yc, bi)
            with tf.GradientTape() as tape:
                loss = lossf(yb, model(xb, training=True))
                if method == 'fedprox':
                    loss += mu_t / 2.0 * tf.add_n([tf.reduce_sum(tf.square(w - r)) for w, r in zip(tv, ref)])
                if method == 'feddyn':
                    loss += da_t / 2.0 * tf.add_n([tf.reduce_sum(tf.square(w - r)) for w, r in zip(tv, ref)])
                    loss -= tf.add_n([tf.reduce_sum(w * h) for w, h in zip(tv, corr)])
            grads = tape.gradient(loss, tv)
            if clip_norm is not None:
                grads, _ = tf.clip_by_global_norm(grads, clip_norm)
            if method == 'scaffold':                                     # control variates track the CLIPPED
                for s_, g in zip(gsum, grads): s_.assign_add(g)          # gradient field; correction added after
                grads = [g + c for g, c in zip(grads, corr)]
            for w, b, g in zip(tv, buf, grads):
                b.assign(mom_t * b + g); w.assign_sub(lr_t * b)
        return nb

    def entropy_of(Xc):
        a = att_model(Xc[:4096], training=False).numpy()
        return float((-(a * np.log(a + 1e-8)).sum(1)).mean())

    gpu = {k: (tf.constant(X[p]), tf.constant(y[p].astype(np.int32))) for k, p in enumerate(parts)}
    hist, weights_log = [], []
    last_pred = None
    t0 = time.time()
    for r in range(1, rounds + 1):
        for rv, w in zip(ref, gw): rv.assign(w)
        local_ws, steps, ent = [], [], []
        for k in range(K):
            for v, w in zip(tv, gw): v.assign(w)                            # every client starts from the global model
            for b in buf: b.assign(tf.zeros_like(b))                         # fresh optimiser state for every client
            for s_ in gsum: s_.assign(tf.zeros_like(s_))
            if method == 'scaffold':
                for cv, cg, cl in zip(corr, c_glob, c_loc[k]): cv.assign(cg - cl)
            if method == 'feddyn':
                for cv, h in zip(corr, dyn_h_loc[k]): cv.assign(h)
            Xc, yc = gpu[k]; nst = 0
            for e in range(local_epochs):
                perm = tf.constant(np.random.default_rng(seed * 100003 + r * 101 + k * 7 + e).permutation(int(n_k[k])), tf.int32)
                nst += int(local_epoch(Xc, yc, perm))
            wk = [v.numpy().copy() for v in tv]; local_ws.append(wk); steps.append(max(nst, 1))
            if method == 'scaffold':
                c_loc[k] = [s_.numpy() / max(nst, 1) for s_ in gsum]       # client control variate = mean raw gradient
            if method == 'feddyn':
                dyn_h_loc[k] = [h - dyn_alpha * (w1 - g0) for h, w1, g0 in zip(dyn_h_loc[k], wk, gw)]
            if method in ('fedadapt', 'fedadapt_pure', 'fedadapt_inv'):
                ent.append(entropy_of(Xc))
        # ---------------- aggregation ----------------
        if method == 'fedadapt':
            a = n_k * np.array(ent); a = a / a.sum()
        elif method == 'fedadapt_pure':
            a = np.array(ent); a = a / a.sum()
        elif method == 'fedadapt_inv':
            a = n_k / np.array(ent); a = a / a.sum()
        elif method == 'fedentropy':
            a = n_k * lab_ent; a = a / a.sum()
        elif method in ('scaffold', 'feddyn'):
            a = np.ones(K) / K                                            # both algorithms average uniformly
        else:
            a = n_k / n_k.sum()
        avg = [sum(a[k] * local_ws[k][L] for k in range(K)) for L in range(len(gw))]
        if method == 'scaffold':
            c_glob = [sum(c_loc[k][L] for k in range(K)) / K for L in range(len(gw))]
            gw = avg
        elif method == 'feddyn':
            dyn_h_srv = [h - dyn_alpha * sum(local_ws[k][L] - gw[L] for k in range(K)) / K
                         for L, h in enumerate(dyn_h_srv)]
            gw = [m - h / dyn_alpha for m, h in zip(avg, dyn_h_srv)]
        elif method == 'fednova':                                   # Wang et al. 2020, SGD with momentum
            tau = np.array(steps, dtype=np.float64); rho = momentum
            a_k = (tau - rho * (1 - rho ** tau) / (1 - rho)) / (1 - rho)
            tau_eff = float((a * a_k).sum())
            gw = [gw[L] - tau_eff * sum(a[k] * (gw[L] - local_ws[k][L]) / a_k[k] for k in range(K)) for L in range(len(gw))]
        else:
            gw = avg
        weights_log.append(a.round(5).tolist())
        if r == 1 or r % eval_every == 0 or r == rounds:
            for v, w in zip(tv, gw): v.assign(w)
            pv = model.predict(Xv, batch_size=8192, verbose=0).argmax(1)
            rec = {'round': r, 'val_macro_f1': float(f1_score(yv, pv, average='macro', labels=np.arange(N_CLASSES)))}
            if Xt is not None:
                pt = model.predict(Xt, batch_size=8192, verbose=0).argmax(1)
                rec['test_macro_f1'] = float(f1_score(yt, pt, average='macro', labels=np.arange(N_CLASSES)))
                rec['test_per_class_f1'] = [float(v) for v in f1_score(yt, pt, average=None, labels=np.arange(N_CLASSES))]
                rec['n_pred_classes'] = int(len(np.unique(pt)))
                if r == rounds: last_pred = pt.astype(np.uint8)
            if method in ('fedadapt', 'fedadapt_pure', 'fedadapt_inv'): rec['entropies'] = [round(e, 5) for e in ent]
            hist.append(rec)
            if log: log(f'    {method} s{seed} r{r:02d} val {rec["val_macro_f1"]:.4f}'
                        + (f' test {rec["test_macro_f1"]:.4f}' if Xt is not None else ''))
            if not all(np.isfinite(w).all() for w in gw): raise FloatingPointError(f'non-finite weights at round {r}')
    out = {'method': method, 'seed': seed, 'rounds': rounds, 'lr': lr, 'momentum': momentum, 'clip_norm': clip_norm, 'mu': mu, 'dyn_alpha': dyn_alpha,
           'local_epochs': local_epochs, 'batch': batch, 'init_sig': init_sig, 'partition_sig': partition_sig(parts),
           'client_sizes': n_k.astype(int).tolist(), 'history': hist, 'agg_weights': weights_log,
           'minutes': (time.time() - t0) / 60, 'pred_test': last_pred}
    del gpu, model, att_model; tf.keras.backend.clear_session(); gc.collect()
    return out
