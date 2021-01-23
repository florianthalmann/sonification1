import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as Tone from 'tone';
import * as _ from 'lodash';

import videoURL from './shibuya.mp4'//'./我的故事.MP4'//'./shibuya2.mp4';
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

const players = {};

async function detect() {
  if (!model) {
    model = await modelPromise;
    console.log('model loaded');
  }
  console.time('predict1');
  const predictions = await model.detect(video);
  console.timeEnd('predict1');
  
  function getRandomColour(){
    const red = Math.floor(Math.random()* 255);
    const green = Math.floor(Math.random() * 255);
    const blue = Math.floor(Math.random() * 255);
    return "rgba("+red+","+green+"," +blue+", 0.5 )";  
  }
  
  context.font = '10px Arial';
  
  context.clearRect(0, 0, clientWidth, clientHeight);
  predictions.forEach(box => {

      const heightScale = (clientHeight / videoHeight);
      const widthScale = (clientWidth / videoWidth);
      
      box.bbox[0] *= widthScale;
      box.bbox[1] = box.bbox[1] * heightScale + 80;
      box.bbox[2] *= widthScale;
      box.bbox[3] *= heightScale;
      
      if (box.bbox[2] < 0.7 * clientWidth) {
        context.lineWidth = 2;
        context.strokeStyle = getRandomColour();
        context.fillStyle = getRandomColour();
        context.fillRect(...box.bbox);
        context.fillStyle = 'white';
        context.fillText(
          box.score.toFixed(3) + ' ' + box.class, box.bbox[0],
          box.bbox[1] > 10 ? box.bbox[1] - 5 : 10);
      }
  })
  
  //update players based on object counts
  const counts = _.countBy(predictions, p => p.class);
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
