import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import { reportUsage, cleanUp } from './util';

const modelFileName = 'ttt_model.json';

(async () => {
  await tf.ready();

  const modelPath = path.join(__dirname, '../models', modelFileName);
  console.log(`Reading model: ${modelPath}`);
  const model = await tf.loadLayersModel(`file://${modelPath}`);

  const emptyBoard = tf.zeros([9]);
  const betterBlockMe = tf.tensor([-1, 0, 0, 1, 1, -1, 0, 0, -1]);
  const goForTheKill = tf.tensor([1, 0, 1, 0, -1, -1, -1, 0, 1]);
  const matches = tf.stack([emptyBoard, betterBlockMe, goForTheKill]);
  const result = model.predict(matches);
  const reshaped = result.reshape([3, 3, 3]);
  console.log('Result tensor:');
  reshaped.print();
  cleanUp(model, emptyBoard, betterBlockMe, goForTheKill, matches, result, reshaped);
  reportUsage();
})();
