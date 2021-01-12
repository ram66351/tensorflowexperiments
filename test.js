import * as tsne from '@tensorflow/tfjs-tsne';

// Create some data
const data = tf.randomUniform([2000,10]);

// Initialize the tsne optimizer
const tsneOpt = tsne.tsne(data);

// Compute a T-SNE embedding, returns a promise.
// Runs for 1000 iterations by default.
tsneOpt.compute().then(() => {
  // tsne.coordinate returns a *tensor* with x, y coordinates of
  // the embedded data.
  const coordinates = tsneOpt.coordinates();
  coordinates.print();
}) ;