// Identifying an image with the Inception model, as explained in chapter 5

import * as tf from '@tensorflow/tfjs-node';
import { reportUsage, cleanUp } from './util';
import { inceptionClasses } from './data';

import path from 'path';
import fs from 'fs';

const modelUrl =
  'https://tfhub.dev/google/tfjs-model/imagenet/inception_v3/classification/3/default/1';
const imageFileName = 'ford-f150.jpg';
// const imageFileName = 'prince.jpg';
// const imageFileName = 'banana.png';

(async () => {
  await tf.ready();

  console.log('Loading Inception');
  const model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });

  const imagePath = path.join(__dirname, '../images', imageFileName);
  console.log(`Reading image: ${imagePath}`);
  const image = fs.readFileSync(imagePath);
  const originalTensor = tf.node.decodeImage(image);
  const preparedTensor = tf.tidy(() =>
    tf.image
      .resizeBilinear(originalTensor, [299, 299], true)
      .div(255)
      .reshape([1, 299, 299, 3])
  );

  console.log('Making predictions');
  const classifications =  model.predict(preparedTensor);
  classifications.print();

  const indices = tf.tidy(() => tf.topk(classifications, 3).indices);
  const winners = indices.dataSync();
  console.log(`
    🥇 First place ${inceptionClasses[winners[0]]} 
    🥈 Second place ${inceptionClasses[winners[1]]}
    🥉 Third place ${inceptionClasses[winners[2]]}
  `);
  cleanUp(originalTensor, preparedTensor, classifications, indices, model);
  reportUsage();
})();
