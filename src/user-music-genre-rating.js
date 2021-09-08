import * as tf from "@tensorflow/tfjs-node";

const users = ["Patrick", "Petra", "Leonore", "Matilda"];
const artists = [
  "Ariana Grande",
  "Prince",
  "Leonard Cohen",
  "Helloween",
  "Aeronauten",
  "Tuxedo Moon",
];
const features = ["Pop", "Funk", "Singer/Songwriter", "Metal", "Indie"];

const userVotes = tf.tensor([
  [3, 10, 10, 7, 5, 6],
  [4, 3, 0, 2, 8, 10],
  [10, 4, 0, 1, 0, 1],
  [5, 3, 0, 10, 4, 2],
]);

const artistFeatures = tf.tensor([
  [1, 0, 0, 0, 0],
  [1, 1, 0, 0, 0],
  [1, 0, 1, 0, 0],
  [0, 0, 0, 1, 0],
  [0, 0, 0, 0, 1],
  [1, 0, 0, 0, 1],
]);

const userFeatures = tf.matMul(userVotes, artistFeatures);

const topUserFeatures = tf.topk(userFeatures, features.length);

const topGenres = topUserFeatures.indices.arraySync();

users.forEach((user, i) => {
  const rankedCategories = topGenres[i].map((v) => features[v]);
  console.log(user, rankedCategories);
});
