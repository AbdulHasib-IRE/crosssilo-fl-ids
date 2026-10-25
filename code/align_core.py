
"""align_core.py - cross-silo FedAvg with optional domain alignment (CORAL, RFF-MMD) or batch normalization (shared / silo-local).
Same clients, initial weights, local optimizer (SGD momentum 0.9 reset per client), clipping and evaluation as the cross-silo grid."""
import time, gc, numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import f1_score, confusion_matrix
NC = 5; LAB = np.arange(NC)
mf1 = lambda y, p: float(f1_score(y, p, average='macro', labels=LAB))
pcf = lambda y, p: [float(v) for v in f1_score(y, p, average=None, labels=LAB)]

def build_bn_model(n_feat):
    inp = layers.Input(shape=(n_feat,), name='x')
    a = layers.Softmax(name='feat_attention')(layers.Dense(n_feat, name='feat_attn_score')(inp))
    x = layers.Lambda(lambda t: t[0] * t[1] * n_feat, name='attn_reweight')([inp, a])
    h = layers.Dense(256)(x); h = layers.BatchNormalization()(h); h = layers.Activation('relu')(h); h = layers.Dropout(0.2)(h)
    h = layers.Dense(128)(h); h = layers.BatchNormalization()(h); h = layers.Activation('relu', name='h128')(h)
    return Model(inp, layers.Dense(NC, activation='softmax')(h))

def hidden_tensor(model, bn):
    if bn: return model.get_layer('h128').output
    d128 = [l for l in model.layers if isinstance(l, layers.Dense) and l.units == 128]; assert len(d128) == 1
    return d128[0].output

def run_align(DATA, fl_xp, setting, method, seed, lr, lam=0.0, bn=False, bn_local=False, rounds=100, batch=256,
              momentum=0.9, clip_norm=1.0, eval_every=10, alpha=0.5, n_rff=512, log=None):
    doms = ('iot', 'ent') if setting == 'xsilo' else (setting,)
    assert method in ('fedavg', 'coral', 'mmd') and (method == 'fedavg' or setting == 'xsilo')
    assert not bn_local or (bn and setting == 'xsilo')
    X = np.concatenate([DATA[d]['X_train'] for d in doms]); y = np.concatenate([DATA[d]['y_train'] for d in doms])
    parts, silo, off = [], [], 0
    for i, d in enumerate(doms):                                   # identical clients to the cross-silo grid
        ps = fl_xp.partition(DATA[d]['y_train'], 'dirichlet', 10, seed, alpha=alpha)
        parts += [p + off for p in ps]; silo += [i] * len(ps); off += len(DATA[d]['y_train'])
    silo = np.array(silo); S = len(doms)
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    model = build_bn_model(X.shape[1]) if bn else fl_xp.build_model(X.shape[1], True)
    tv = model.trainable_variables; allw = model.weights
    if not bn: assert len(allw) == len(tv) == 8
    tv_idx = [next(j for j, w in enumerate(allw) if w is v) for v in tv]
    bn_ids = {id(w) for l in model.layers if isinstance(l, layers.BatchNormalization) for w in l.weights}
    local = {j for j, w in enumerate(allw) if id(w) in bn_ids} if bn_local else set()
    K = len(parts); n_k = np.array([len(p) for p in parts], dtype=np.float64)
    gw = [w.numpy().copy() for w in allw]; init_sig = fl_xp.wsig([gw[j] for j in tv_idx])
    sw = {i: [w.copy() for w in gw] for i in range(S)}
    buf = [tf.Variable(np.zeros_like(v.numpy()), trainable=False) for v in tv]
    lossf = tf.keras.losses.SparseCategoricalCrossentropy()
    lr_t, mom_t, lam_t = tf.constant(lr, tf.float32), tf.constant(momentum, tf.float32), tf.constant(lam, tf.float32)
    feat_model = Model(model.input, [hidden_tensor(model, bn), model.output])
    H = 128
    cov_o = tf.Variable(np.zeros((H, H), np.float32), trainable=False); emb_o = tf.Variable(np.zeros(n_rff, np.float32), trainable=False)
    rng = np.random.default_rng(seed * 7919 + 17); sigma = None; Wr = br = None

    @tf.function(reduce_retracing=True)
    def local_epoch(Xc, yc, perm):                                 # identical to the cross-silo grid step
        nb = tf.shape(perm)[0] // batch
        for i in tf.range(nb):
            bi = perm[i * batch:(i + 1) * batch]; xb, yb = tf.gather(Xc, bi), tf.gather(yc, bi)
            with tf.GradientTape() as tape:
                loss = lossf(yb, model(xb, training=True))
            grads = tape.gradient(loss, tv)
            if clip_norm is not None: grads, _ = tf.clip_by_global_norm(grads, clip_norm)
            for w, b, g in zip(tv, buf, grads): b.assign(mom_t * b + g); w.assign_sub(lr_t * b)
        return nb

    def coral_pen(h):
        n = tf.cast(tf.shape(h)[0], tf.float32); hc = h - tf.reduce_mean(h, 0, keepdims=True)
        c = tf.matmul(hc, hc, transpose_a=True) / (n - 1.0)
        return tf.reduce_sum(tf.square(c - cov_o)) / (4.0 * H * H)
    def mmd_pen(h, Wt, bt):
        phi = tf.sqrt(2.0 / n_rff) * tf.cos(tf.matmul(h, Wt) + bt)
        return tf.reduce_sum(tf.square(tf.reduce_mean(phi, 0) - emb_o))

    def make_pen_epoch(Wt, bt):
        @tf.function(reduce_retracing=True)
        def pen_epoch(Xc, yc, perm):
            nb = tf.shape(perm)[0] // batch; tot = tf.constant(0.0)
            for i in tf.range(nb):
                bi = perm[i * batch:(i + 1) * batch]; xb, yb = tf.gather(Xc, bi), tf.gather(yc, bi)
                with tf.GradientTape() as tape:
                    h, p = feat_model(xb, training=True)
                    pen = coral_pen(h) if method == 'coral' else mmd_pen(h, Wt, bt)
                    loss = lossf(yb, p) + lam_t * pen
                grads = tape.gradient(loss, tv)
                if clip_norm is not None: grads, _ = tf.clip_by_global_norm(grads, clip_norm)
                for w, b, g in zip(tv, buf, grads): b.assign(mom_t * b + g); w.assign_sub(lr_t * b)
                tot += pen
            return tot / tf.cast(tf.maximum(nb, 1), tf.float32)
        return pen_epoch

    def silo_stats(i):                                             # aggregate statistics of silo i under silo i's model
        for v, w in zip(allw, sw[i]): v.assign(w)
        h = feat_model.predict(DATA[doms[i]]['X_train'], batch_size=8192, verbose=0)[0].astype(np.float64)
        out = {'cov': np.cov(h, rowvar=False).astype(np.float32)}
        if method == 'mmd': out['emb'] = (np.sqrt(2.0 / n_rff) * np.cos(h @ Wr + br)).mean(0).astype(np.float32)
        return out, h

    XV = np.concatenate([DATA[d]['X_val'] for d in doms]); XT = np.concatenate([DATA[d]['X_test'] for d in doms])
    CV = np.cumsum([0] + [len(DATA[d]['y_val']) for d in doms]); CT = np.cumsum([0] + [len(DATA[d]['y_test']) for d in doms])
    gpu = {k: (tf.constant(X[p]), tf.constant(y[p].astype(np.int32))) for k, p in enumerate(parts)}
    hist, pens = [], []; t0 = time.time(); pen_epoch = None
    if method == 'mmd':                                            # bandwidth: mean of each silo's own median pairwise distance (initial model)
        meds = []
        for i in range(S):
            for v, w in zip(allw, sw[i]): v.assign(w)
            h = feat_model.predict(DATA[doms[i]]['X_train'][rng.choice(len(DATA[doms[i]]['X_train']), 1000, replace=False)], verbose=0)[0]
            dd = np.sqrt(((h[:, None, :] - h[None, :, :]) ** 2).sum(-1)); meds.append(float(np.median(dd[np.triu_indices(1000, 1)])))
        sigma = max(float(np.mean(meds)), 1e-3)
        Wr = (rng.standard_normal((H, n_rff)) / sigma).astype(np.float32); br = rng.uniform(0, 2 * np.pi, n_rff).astype(np.float32)
    if method != 'fedavg': pen_epoch = make_pen_epoch(tf.constant(Wr if Wr is not None else np.zeros((H, n_rff), np.float32)),
                                                       tf.constant(br if br is not None else np.zeros(n_rff, np.float32)))
    for r in range(1, rounds + 1):
        st = [silo_stats(i)[0] for i in range(S)] if method != 'fedavg' else None
        local_ws, rp = [], []
        for k in range(K):
            for v, w in zip(allw, sw[silo[k]]): v.assign(w)
            for b in buf: b.assign(tf.zeros_like(b))
            Xc, yc = gpu[k]
            perm = tf.constant(np.random.default_rng(seed * 100003 + r * 101 + k * 7 + 0).permutation(int(n_k[k])), tf.int32)
            if method == 'fedavg': local_epoch(Xc, yc, perm)
            else:
                o = 1 - silo[k]; cov_o.assign(st[o]['cov'])
                if method == 'mmd': emb_o.assign(st[o]['emb'])
                rp.append(float(pen_epoch(Xc, yc, perm)))
            local_ws.append([v.numpy().copy() for v in allw])
        a = n_k / n_k.sum()
        shared = [sum(a[k] * local_ws[k][L] for k in range(K)) for L in range(len(allw))]
        for i in range(S):
            if local:
                ai = np.where(silo == i, n_k, 0.0); ai = ai / ai.sum()
                loc = [sum(ai[k] * local_ws[k][L] for k in range(K)) for L in range(len(allw))]
                sw[i] = [loc[L] if L in local else shared[L] for L in range(len(allw))]
            else: sw[i] = [w for w in shared]
        if rp: pens.append(float(np.mean(rp)))
        if r == 1 or r % eval_every == 0 or r == rounds:
            rec = {'round': r}
            if rp: rec['penalty'] = pens[-1]
            for i, d in enumerate(doms):
                for v, w in zip(allw, sw[i]): v.assign(w)
                pv = model.predict(XV, batch_size=8192, verbose=0).argmax(1)[CV[i]:CV[i + 1]]
                pt = model.predict(XT, batch_size=8192, verbose=0).argmax(1)[CT[i]:CT[i + 1]]
                rec[f'val_{d}'] = mf1(DATA[d]['y_val'], pv); rec[f'test_{d}'] = mf1(DATA[d]['y_test'], pt)
                rec[f'test_pc_{d}'] = pcf(DATA[d]['y_test'], pt); rec[f'npred_{d}'] = int(len(np.unique(pt)))
                if r == rounds: rec[f'cm_{d}'] = confusion_matrix(DATA[d]['y_test'], pt, labels=LAB).tolist()
            hist.append(rec)
            if log: log(r, rec)
            if not all(np.isfinite(w).all() for i in range(S) for w in sw[i]): return {'diverged': True, 'history': hist}
    del gpu, model, feat_model; tf.keras.backend.clear_session(); gc.collect()
    return {'diverged': False, 'history': hist, 'minutes': (time.time() - t0) / 60, 'init_sig': init_sig,
            'partition_sig': fl_xp.partition_sig(parts), 'client_sizes': n_k.astype(int).tolist(), 'setting': setting,
            'method': method, 'lam': lam, 'bn': bn, 'bn_local': bn_local, 'lr': lr, 'seed': seed, 'sigma': sigma,
            'penalty_per_round': pens, 'n_local_tensors': len(local)}
