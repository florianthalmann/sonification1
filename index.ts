import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as Tone from 'tone';
import * as _ from 'lodash';

import videoURL from './H3.mov'//'./shibuya.mp4'//'./我的故事.MP4'//'./shibuya2.mp4';
import sounds from './sounds/*.mp3';

let modelPromise: Promise<cocoSsd.ObjectDetection>;

let context: CanvasRenderingContext2D;
let video: HTMLVideoElement;
let videoHeight = 0, videoWidth = 0, clientHeight = 0, clientWidth = 0;
let model: cocoSsd.ObjectDetection;

Tone.start();

window.onload = () => {
  modelPromise = cocoSsd.load();//{base: "mobilenet_v1"});
  
  video = <HTMLVideoElement>document.getElementById('video');
  video.src = videoURL;
  
  const canvas = <HTMLCanvasElement>document.getElementById('canvas');
  context = canvas.getContext('2d');
  canvas.innerHTML = ""
  
  document.getElementById("everything").addEventListener("click", function() {
    Tone.start();
  });
  
  video.addEventListener("canplay", () => {
    videoHeight = video.videoHeight;
    videoWidth = video.videoWidth;
    clientHeight = video.clientHeight;
    clientWidth = video.clientWidth;
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    detect();
  })
}

interface Object {
  bbox: [number, number, number, number],
  class: string,
  score: number,
  color: string
}

let objects: Object[] = [];
const players = {};

async function detect() {
  if (!model) {
    model = await modelPromise;
    console.log('model loaded');
  }
  console.time('predict1');
  const predictions = await model.detect(video);
  console.timeEnd('predict1');
  
  context.font = '10px Arial';
  
  //pair up objects and update
  const pairs = pairUp(objects, predictions);
  pairs.forEach(([i,j]) => _.extend(objects[i], predictions[j]));
  //remove objects that disappeared and add new objects
  objects = objects.filter((_o,i) => pairs.map(p => p[0]).indexOf(i) >= 0);
  _.difference(_.range(predictions.length), pairs.map(p => p[1])).forEach(j =>
    objects.push(_.extend(predictions[j], {color: getRandomColor()})));
  
  const heightScale = (clientHeight / videoHeight);
  const widthScale = (clientWidth / videoWidth);
  
  context.clearRect(0, 0, clientWidth, clientHeight);
  objects.forEach(object => {
      
      const scaledBox: [number, number, number, number] = [
        object.bbox[0] * widthScale,
        object.bbox[1] * heightScale + 80,
        object.bbox[2] * widthScale,
        object.bbox[3] * heightScale
      ]
      
      if (scaledBox[2] < 0.7 * clientWidth) {
        context.lineWidth = 2;
        context.strokeStyle = object.color//getRandomColor();
        context.fillStyle = object.color//getRandomColor();
        context.fillRect(...scaledBox);
        context.fillStyle = 'white';
        context.fillText(
          object.score.toFixed(3) + ' ' + object.class, scaledBox[0],
          scaledBox[1] > 10 ? scaledBox[1] - 5 : 10);
      }
  })
  
  //update players based on object counts
  const counts = _.countBy(objects, p => p.class);
  console.log(counts)
  
  for (let c in counts) {
    const csounds = _.values(sounds).filter(s => s.indexOf(c) >= 0);
    if (csounds.length > 0) {
      if (!players[c]) players[c] = [];
      
      while (players[c].length < _.ceil(counts[c]/2)) {
        const player = new Tone.Player(_.sample(csounds)).toDestination();
        player.autostart = true;
        players[c].push(player);
      }
      
      while (players[c].length > _.ceil(counts[c]/2)) {
        players[c].pop().stop();
      }
    }
  }
  
  setTimeout(detect, 10);
};

function getRandomColor() {
  const red = Math.floor(Math.random()* 255);
  const green = Math.floor(Math.random() * 255);
  const blue = Math.floor(Math.random() * 255);
  return "rgba("+red+","+green+"," +blue+", 0.5 )";  
}

function pairUp(o1: cocoSsd.DetectedObject[], o2: cocoSsd.DetectedObject[]) {
  const dists: [number, number, number][][] = o1.map((o,i) => o2.map((p,j) =>
    [i, j, euclideanDist(o.bbox.slice(0,2), p.bbox.slice(0,2))]));
  const mins = _.sortBy(_.flatten(dists), d => d[2]);
  //console.log(o1, o2, dists, mins, _.min([o1.length, o2.length]))
  const pairs: [number, number][] = [];
  while (pairs.length < _.min([o1.length, o2.length])) {
    const [ii, jj] = pairs.length > 0 ? _.unzip(pairs) : [[],[]];
    //console.log(pairs, ii, jj)
    const min = mins.filter(([i,j,_d]) => ii.indexOf(i) < 0 && jj.indexOf(j) < 0)[0]
    pairs.push([min[0], min[1]]);
  }
  return pairs;
}

function euclideanDist(v1: number[], v2: number[]) {
  return Math.sqrt(_.sum(_.range(v1.length).map(i =>
      Math.pow(Math.abs(v1[i] - v2[i]), 2))));
}
