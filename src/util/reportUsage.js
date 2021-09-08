import * as tf from '@tensorflow/tfjs-node';

export default () => {
  console.log('Number of tensors:', tf.memory().numTensors);
  console.log('Allocated memory (bytes):', tf.memory().numBytes);
};
