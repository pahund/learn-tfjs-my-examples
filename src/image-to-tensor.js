import * as tf from '@tensorflow/tfjs-node-gpu';
import * as fs from 'fs';
import * as path from 'path';

const imagePath = path.join(__dirname, '../images/truck.jpg');
console.log(`Reading image: ${imagePath}`);
const image = fs.readFileSync(imagePath);

tf.tidy(() => {
  const tensor = tf.node.decodeImage(image);
  console.log(`Success: local file was converted to a ${tensor.shape} tensor`)
})

