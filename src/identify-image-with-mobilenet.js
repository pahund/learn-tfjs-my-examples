import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import * as mobilenet from '@tensorflow-models/mobilenet';

// const imageFileName = 'ford-f150.jpg';
// const imageFileName = 'prince.jpg';
const imageFileName = 'banana.png';

(async () => {
  console.log('Loading mobilenet');
  const model = await mobilenet.load();

  const imagePath = path.join(__dirname, '../images', imageFileName);
  console.log(`Reading image: ${imagePath}`);
  const image = fs.readFileSync(imagePath);
  const tensor = tf.node.decodeImage(image);

  console.log('Making predictions')
  const predictions = await model.classify(tensor);

  console.log(`Predictions:\n${JSON.stringify(predictions, null, 4)}`); // PH_TODO

  tensor.dispose();
  model.dispose();
})();
