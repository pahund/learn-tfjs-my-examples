import * as tf from "@tensorflow/tfjs-node";

const lil = tf.tensor([
  [[1], [0]],
  [[0], [1]],
]);

const big = lil.tile([10, 10,1]);

big.print();
