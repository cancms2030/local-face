import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/vision_bundle.mjs";

// ========= 配置常量 =========
const MP_MODEL = './models/face_landmarker.task';
const ONNX_MODEL = './models/mobilefacenet.onnx';
const SIM_THRESHOLD = 0.5; // 比对相似度阈值
const STORAGE_KEY = "face_local_users";

// ArcFace标准5点目标坐标
const ARCFACE_DST = [
  [38.2946, 51.6963], // 左眼
  [73.5318, 51.5014], // 右眼
  [56.0252, 71.7366], // 鼻尖
  [41.5493, 92.3655], // 左嘴角
  [70.7299, 92.2041], // 右嘴角
];

// ========= DOM 元素 =========
const video = document.getElementById('video');
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');
const alignCanvas = document.getElementById('alignCanvas');
const alignCtx = alignCanvas.getContext('2d', { willReadFrequently: true });
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const userInfoEl = document.getElementById('userInfo');
const inputUsername = document.getElementById('inputUsername');
const btnRegister = document.getElementById('btnRegister');
const btnLogin = document.getElementById('btnLogin');
const btnClearLocal = document.getElementById('btnClearLocal');
const fileUpload = document.getElementById('fileUpload');
const uploadedImg = document.getElementById('uploadedImg');

// ========= 全局状态 =========
let faceLandmarker = null;
let ortSession = null;
let lastLandmarks = null;
let isProcessing = false;

// ========= 数学工具：仿射变换求解 =========
function solveAffine(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < src.length; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0]); b.push(u);
    A.push([0, 0, 0, x, y, 1]); b.push(v);
  }
  const AT = A[0].map((_, i) => A.map(row => row[i]));
  const ATA = Array(6).fill(0).map(() => Array(6).fill(0));
  const ATb = Array(6).fill(0);
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) {
      ATA[i][j] = AT[i].reduce((s, val, k) => s + val * A[k][j], 0);
    }
    ATb[i] = AT[i].reduce((s, val, k) => s + val * b[k], 0);
  }
  const M = ATA.map((row, i) => [...row, ATb[i]]);
  for (let col = 0; col < 6; col++) {
    let pivot = col;
    for (let r = col + 1; r < 6; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    [M[col], M[pivot]] = [M[pivot], M[col]];
    const pv = M[col][col];
    for (let j = col; j <= 6; j++) M[col][j] /= pv;
    for (let r = 0; r < 6; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let j = col; j <= 6; j++) M[r][j] -= factor * M[col][j];
    }
  }
  return M.map(row => row[6]);
}

// 人脸对齐
function alignFace(source, landmarks, canvasCtx) {
  const W = source.videoWidth || source.naturalWidth || source.width;
  const H = source.videoHeight || source.naturalHeight || source.height;
  const getPx = (idx) => [landmarks[idx].x * W, landmarks[idx].y * H];
  const srcPoints = [
    getPx(33), getPx(263), getPx(1), getPx(61), getPx(291)
  ];
  const M = solveAffine(srcPoints, ARCFACE_DST);
  canvasCtx.fillStyle = "#000";
  canvasCtx.fillRect(0, 0, 112, 112);
  canvasCtx.setTransform(M[0], M[1], M[3], M[4], M[2], M[5]);
  canvasCtx.drawImage(source, 0, 0);
  canvasCtx.setTransform(1, 0, 0, 1, 0, 0);
}

// 生成简单UUID作为用户唯一ID
function genUserId() {
  return crypto.randomUUID();
}

// 本地存储工具
function getLocalUsers(){
  const str = localStorage.getItem(STORAGE_KEY);
  if(!str) return [];
  try{
    return JSON.parse(str);
  }catch(e){
    return [];
  }
}
function saveLocalUser(userObj){
  const list = getLocalUsers();
  list.push(userObj);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}
function clearAllLocalUser(){
  localStorage.removeItem(STORAGE_KEY);
}

// 提取人脸特征向量
async function getEmbedding() {
  const imgData = alignCtx.getImageData(0, 0, 112, 112).data;
  const float32Data = new Float32Array(1 * 3 * 112 * 112);
  for (let i = 0; i < 112 * 112; i++) {
    float32Data[i] = (imgData[i * 4] - 127.5) / 128.0;
    float32Data[112 * 112 + i] = (imgData[i * 4 + 1] - 127.5) / 128.0;
    float32Data[2 * 112 * 112 + i] = (imgData[i * 4 + 2] - 127.5) / 128.0;
  }
  const tensor = new ort.Tensor('float32', float32Data, [1, 3, 112, 112]);
  const feeds = { "input.1": tensor };
  const results = await ortSession.run(feeds);
  return Array.from(results["516"].data);
}

// 余弦相似度
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// 人脸比对：遍历本地用户库，返回匹配到的用户 || null
function matchUser(currentVec){
  const users = getLocalUsers();
  for(const u of users){
    const sim = cosineSimilarity(currentVec, u.embedding);
    if(sim >= SIM_THRESHOLD){
      return {...u, similarity: sim};
    }
  }
  return null;
}

// ========= 初始化引擎 =========
async function initMediaPipe() {
  statusEl.textContent = "加载 MediaPipe...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MP_MODEL },
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

async function initONNX() {
  statusEl.textContent = "加载 MobileFaceNet...";
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
  const buffer = await fetch(ONNX_MODEL).then(r => r.arrayBuffer());
  ortSession = await ort.InferenceSession.create(buffer, {
    executionProviders: ['wasm']
  });
}

async function initCamera() {
  statusEl.textContent = "启动摄像头...";
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: 640, height: 480 }
  });
  video.srcObject = stream;
  return new Promise(resolve => video.onloadedmetadata = resolve);
}

// 摄像头循环
async function detectLoop() {
  if (video.readyState === 4 && !isProcessing) {
    isProcessing = true;
    try {
      const results = faceLandmarker.detectForVideo(video, performance.now());
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        lastLandmarks = results.faceLandmarks[0];
        alignFace(video, lastLandmarks, alignCtx);
        previewCtx.drawImage(alignCanvas, 0, 0);
      } else {
        lastLandmarks = null;
      }
    } catch (e) { console.error(e); }
    isProcessing = false;
  }
  requestAnimationFrame(detectLoop);
}

// ========= 按钮事件 =========
btnRegister.addEventListener('click', async ()=>{
  const username = inputUsername.value.trim();
  if(!username){
    alert("请输入用户名");
    return;
  }
  if(!lastLandmarks){
    alert("没有检测到人脸，请正对摄像头或者上传人脸图片");
    return;
  }
  resultEl.textContent = "正在提取人脸特征向量...";
  userInfoEl.innerText = "";
  try{
    const embedding = await getEmbedding();
    const userId = genUserId();
    const userItem = {
      userId,
      username,
      embedding // 数组会被JSON序列化存入localStorage
    };
    saveLocalUser(userItem);
    resultEl.innerHTML = `<span class="success">✅ 注册成功，数据保存在浏览器本地</span>`;
    userInfoEl.innerText = `用户ID:${userId}\n用户名:${username}`;
  }catch(err){
    resultEl.innerHTML = `<span class="error">注册失败:${err.message}</span>`;
  }
});

btnLogin.addEventListener('click', async ()=>{
  if(!lastLandmarks){
    alert("未检测人脸，请对准摄像头或上传图片");
    return;
  }
  resultEl.textContent = "正在人脸比对...";
  userInfoEl.innerText = "";
  try{
    const currVec = await getEmbedding();
    const matched = matchUser(currVec);
    if(matched){
      resultEl.innerHTML = `<span class="success">✅ 人脸认证通过，相似度:${matched.similarity.toFixed(3)}</span>`;
      userInfoEl.innerText = `匹配成功用户ID：${matched.userId}\n用户名：${matched.username}\n相似度：${matched.similarity.toFixed(3)}`;
    }else{
      resultEl.innerHTML = `<span class="error">❌ 人脸认证失败，没有匹配本地注册用户</span>`;
    }
  }catch(err){
    resultEl.innerHTML = `<span class="error">比对异常：${err.message}</span>`;
  }
});

btnClearLocal.addEventListener('click',()=>{
  if(confirm("确认清空浏览器本地全部注册人脸用户？不可恢复！")){
    clearAllLocalUser();
    resultEl.textContent = "已清空本地存储";
    userInfoEl.innerText = "";
  }
});

// 图片上传测试
fileUpload.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    uploadedImg.src = event.target.result;
    uploadedImg.style.display = 'block';
    uploadedImg.onload = () => {
      if (!faceLandmarker) return alert('引擎还未初始化完成');
      const results = faceLandmarker.detectForVideo(uploadedImg, performance.now());
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        alignFace(uploadedImg, results.faceLandmarks[0], alignCtx);
        previewCtx.drawImage(alignCanvas, 0, 0);
        lastLandmarks = results.faceLandmarks[0];
        statusEl.textContent = "图片人脸就绪，可以注册/登录";
        statusEl.style.color="#4ade80";
      } else {
        statusEl.textContent = "图片没有检测到人脸";
        statusEl.style.color="#f87171";
      }
    };
  };
  reader.readAsDataURL(file);
});

// ========= 启动入口 =========
(async ()=>{
  try{
    if (typeof ort === 'undefined') throw new Error("onnxruntime 未加载");
    await initMediaPipe();
    await initONNX();
    let camOk = false;
    try{
      await initCamera();
      detectLoop();
      camOk = true;
    }catch(err){
      console.warn("摄像头启动失败，仅图片上传模式",err);
      statusEl.textContent = "摄像头不可用，请上传图片测试";
    }
    if(camOk){
      statusEl.textContent = "引擎就绪，请录入人脸，输入用户名注册";
    }
    statusEl.style.color="#4ade80";
    btnRegister.disabled = false;
    btnLogin.disabled = false;
  }catch(e){
    statusEl.textContent = "初始化失败:"+e.message;
    statusEl.style.color="#f87171";
    console.error(e);
  }
})();
