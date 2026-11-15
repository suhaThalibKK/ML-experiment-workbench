export const softmax = scores => {
  const max = Math.max(...scores), values = scores.map(x => Math.exp(x - max));
  const total = values.reduce((a, b) => a + b, 0);
  return values.map(x => x / total);
};
export function predictID3(tree, sample) {
  let node = tree; const path = [];
  while (node.attribute !== null) {
    const rawValue = sample[node.attribute];
    const value = typeof rawValue === "boolean" ? (rawValue ? "True" : "False") : String(rawValue);
    path.push({ attribute: node.attribute, value });
    if (!Object.hasOwn(node.children, value)) return { label: node.prediction, path, fallback: true };
    node = node.children[value];
  }
  return { label: node.prediction, path, fallback: false };
}
export function predictGaussian(model, inputs) {
  if (inputs.length !== model.features.length || inputs.some(x => !Number.isFinite(x))) throw new Error('Enter a numeric value for every feature.');
  const scores = model.classes.map((_, i) => model.logPriors[i] - 0.5 * inputs.reduce((sum, x, j) => {
    const variance = model.variances[i][j];
    return sum + Math.log(2 * Math.PI * variance) + (x - model.means[i][j]) ** 2 / variance;
  }, 0));
  const probabilities = softmax(scores);
  return { label: model.classes[probabilities.indexOf(Math.max(...probabilities))], probabilities, scores };
}
export function predictDocument(model, message) {
  const words = message.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || [];
  const counts = new Map();
  for (const word of words) if (Object.hasOwn(model.vocabulary, word)) counts.set(word, (counts.get(word) || 0) + 1);
  const scores = model.classes.map((_, i) => {
    let score = model.classLogPrior[i];
    for (const [word, count] of counts) score += count * model.featureLogProb[i][model.vocabulary[word]];
    return score;
  });
  const probabilities = softmax(scores);
  return { label: model.classes[probabilities.indexOf(Math.max(...probabilities))], probabilities,
    matchedWords: Array.from(counts.keys()), unknown: words.length - [...counts.values()].reduce((a, b) => a + b, 0) };
}
export function predictHeart(model, raw) {
  const evidence = {};
  for (const feature of model.features) {
    const rule = model.rules[feature];
    const input = raw[feature];
    const value = input === null || input === undefined || input === "" || input === "?"
      ? (rule.median ?? rule.mode) : Number(input);
    if (!Number.isFinite(value)) throw new Error('Enter valid values for every feature.');
    evidence[feature] = rule.cut_points ? rule.cut_points.filter(x => value >= x).length : value;
    if (!model.states[feature].includes(evidence[feature])) throw new Error(`Invalid category for ${feature}.`);
  }
  const scores = [0, 1].map(target => {
    const row = { ...evidence, target };
    return model.cpds.reduce((total, cpd) => {
      let index = 0;
      cpd.variables.forEach((variable, i) => { index = index * cpd.cardinality[i] + cpd.states[variable].indexOf(row[variable]); });
      return total + Math.log(cpd.values[index]);
    }, 0);
  });
  const probabilities = softmax(scores);
  return { label: probabilities[1] > probabilities[0] ? 1 : 0, probabilities, evidence };
}
export function predictKNN(model, raw, k = 3) {
  if (raw.length !== model.mean.length || raw.some(x => !Number.isFinite(x))) throw new Error('Enter all four numeric measurements.');
  if (!Number.isInteger(k) || k < 1 || k > model.trainX.length) throw new Error('Choose a valid number of neighbors.');
  const point = raw.map((x, i) => (x - model.mean[i]) / model.scale[i]);
  const neighbors = model.trainX.map((row, i) => ({ row: model.trainRows[i] + 1, index: i, label: model.trainY[i],
    distance: Math.sqrt(row.reduce((total, value, j) => total + (value - point[j]) ** 2, 0)) }))
    .sort((a, b) => a.distance - b.distance || a.index - b.index).slice(0, k);
  const classes = [...new Set(model.trainY)].sort(), votes = Object.fromEntries(classes.map(c => [c, 0]));
  neighbors.forEach(n => votes[n.label]++);
  const label = classes.reduce((best, candidate) => votes[candidate] > votes[best] ? candidate : best, classes[0]);
  return { label, neighbors, votes };
}
export function multiplyMatrix(values) {
  const [a, b, c, d] = values;
  return [a * a + b * c, a * b + b * d, c * a + d * c, c * b + d * d];
}
