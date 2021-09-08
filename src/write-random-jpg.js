import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";

const bigMess = tf.randomUniform([400, 400, 3], 0, 255);
tf.node.encodeJpeg(bigMess).then((f) => {
  fs.writeFileSync("simple.jpg", f);
  console.log("Basic JPG 'simple.jpg' written");
});
