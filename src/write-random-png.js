import * as tf from "@tensorflow/tfjs-node-gpu";
import fs from "fs";

const bigMess = tf.randomUniform([400, 400, 3], 0, 255);
tf.node.encodePng(bigMess).then((f) => {
  fs.writeFileSync("simple.png", f);
  console.log("Basic JPG 'simple.png' written");
});
