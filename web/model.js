// Rebuilds the Wide & Deep CNN from src/model.py in TensorFlow.js and loads the
// trained weights exported by web/export_for_web.py. Dropout is omitted: it does
// nothing at prediction time.
/* global tf */

function decode({ shape, data }) {
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  return tf.tensor(new Float32Array(bytes.buffer), shape);
}

export async function loadModel(url = "model/weights.json") {
  const cfg = await (await fetch(url)).json();
  const { n_days, n_weeks, n_feats, units, filters, bn_epsilon: epsilon } = cfg;
  const L = {
    conv1: tf.layers.conv2d({ filters, kernelSize: 3, padding: "same", activation: "relu" }),
    bn_deep: tf.layers.batchNormalization({ epsilon }),
    conv2: tf.layers.conv2d({ filters, kernelSize: 3, padding: "same", activation: "relu" }),
    conv3: tf.layers.conv2d({ filters: filters * 2, kernelSize: 3, padding: "same", activation: "relu" }),
    wide: tf.layers.dense({ units, activation: "relu" }),
    bn_stats: tf.layers.batchNormalization({ epsilon }),
    stats: tf.layers.dense({ units: 32, activation: "relu" }),
    hidden: tf.layers.dense({ units, activation: "relu" }),
    out: tf.layers.dense({ units: 1, activation: "sigmoid" }),
  };

  const dailyIn = tf.input({ shape: [n_days] });
  const weeklyIn = tf.input({ shape: [n_weeks, 7, 1] });
  const statsIn = tf.input({ shape: [n_feats] });

  const w = L.wide.apply(dailyIn);
  let d = L.conv1.apply(weeklyIn);
  d = L.bn_deep.apply(d);
  d = L.conv2.apply(d);
  d = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(d);
  d = L.conv3.apply(d);
  d = tf.layers.globalAveragePooling2d({}).apply(d);
  let s = L.bn_stats.apply(statsIn);
  s = L.stats.apply(s);
  let x = tf.layers.concatenate().apply([w, d, s]);
  x = L.hidden.apply(x);
  const out = L.out.apply(x);
  const model = tf.model({ inputs: [dailyIn, weeklyIn, statsIn], outputs: out });

  for (const [name, layer] of Object.entries(L)) layer.setWeights(cfg.weights[name].map(decode));
  return { model, cfg };
}

export function predict(model, inputs, n, cfg) {
  return tf.tidy(() => {
    const p = model.predict([
      tf.tensor2d(inputs.daily, [n, cfg.n_days]),
      tf.tensor4d(inputs.weekly, [n, cfg.n_weeks, 7, 1]),
      tf.tensor2d(inputs.stats, [n, cfg.n_feats]),
    ]);
    return Array.from(p.dataSync());
  });
}
