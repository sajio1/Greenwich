import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const wait = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

async function preparePreview(assetId, isCurrent) {
  let response = await requestJson('/api/prepare', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: assetId, async: true}),
  });
  while (response.status !== 'ready') {
    if (!isCurrent()) throw new Error('Preview request superseded');
    if (response.status === 'failed') throw new Error(response.error || 'Preview decoding failed');
    await wait(450);
    response = await requestJson(`/api/preview?id=${encodeURIComponent(assetId)}`);
  }
  return response.preview;
}

function disposeObject(object) {
  object?.traverse(node => {
    node.geometry?.dispose?.();
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of materials) {
      if (!material) continue;
      for (const value of Object.values(material)) if (value?.isTexture) value.dispose();
      material.dispose?.();
    }
  });
}

class MotionPane {
  constructor(canvas, loading, error, fpsLabel) {
    this.canvas = canvas;
    this.loading = loading;
    this.error = error;
    this.fpsLabel = fpsLabel;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.001, 10000);
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true, powerPreference: 'high-performance'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.scene.add(new THREE.HemisphereLight(0xe7f0ff, 0x1d2633, 2.5));
    const key = new THREE.DirectionalLight(0xffffff, 2.7);
    key.position.set(4, 6, 4);
    this.scene.add(key);
    const grid = new THREE.GridHelper(30, 60, 0x465160, 0x252c35);
    grid.material.opacity = 0.42;
    grid.material.transparent = true;
    this.scene.add(grid);
    this.root = null;
    this.helper = null;
    this.mixer = null;
    this.duration = 0;
    this.setTheme();
  }

  setTheme() {
    this.scene.background = new THREE.Color(document.body.dataset.theme === 'light' ? 0xeef0f2 : 0x202329);
  }

  clear() {
    if (this.root) {
      this.scene.remove(this.root);
      disposeObject(this.root);
    }
    if (this.helper) {
      this.scene.remove(this.helper);
      this.helper.geometry?.dispose?.();
      this.helper.material?.dispose?.();
    }
    this.root = null;
    this.helper = null;
    this.mixer = null;
    this.duration = 0;
  }

  async load(asset, isCurrent) {
    this.clear();
    this.loading.hidden = false;
    this.error.hidden = true;
    this.fpsLabel.textContent = '— FPS';
    try {
      const preview = await preparePreview(asset.id, isCurrent);
      if (!isCurrent()) return;
      if (!preview.model_url || !['glb', 'gltf'].includes(preview.model_format)) {
        throw new Error('Comparison view currently requires a GLB/GLTF preview');
      }
      const gltf = await new GLTFLoader().loadAsync(preview.model_url);
      if (!isCurrent()) {
        disposeObject(gltf.scene);
        return;
      }
      this.root = gltf.scene;
      this.scene.add(this.root);
      let hasBones = false;
      this.root.traverse(node => {
        if (node.isBone) hasBones = true;
        if (node.isMesh) {
          node.castShadow = true;
          node.receiveShadow = true;
        }
      });
      if (hasBones) {
        this.helper = new THREE.SkeletonHelper(this.root);
        this.helper.material.depthTest = false;
        this.helper.material.color.set(0xffa263);
        this.helper.renderOrder = 30;
        this.scene.add(this.helper);
      }
      const animations = gltf.animations || [];
      if (animations.length) {
        this.duration = animations[0].duration;
        this.mixer = new THREE.AnimationMixer(this.root);
        this.mixer.clipAction(animations[0]).play();
        this.mixer.setTime(0);
      }
      const metadata = preview.preview_metadata || {};
      const fps = metadata.source_fps ?? metadata.fps ?? preview.source_fps ?? preview.fps;
      this.fpsLabel.textContent = fps ? `${Number(fps).toFixed(2)} FPS` : 'FPS unknown';
      this.fitCamera();
      this.loading.hidden = true;
    } catch (error) {
      if (!isCurrent() || error.message === 'Preview request superseded') return;
      this.loading.hidden = true;
      this.error.textContent = error.message;
      this.error.hidden = false;
      throw error;
    }
  }

  fitCamera() {
    if (!this.root) return;
    this.root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(this.root);
    if (box.isEmpty()) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const radius = Math.max(sphere.radius, 0.25);
    this.controls.target.copy(sphere.center);
    this.camera.near = Math.max(radius / 1000, 0.001);
    this.camera.far = Math.max(radius * 100, 100);
    this.camera.position.copy(sphere.center).add(new THREE.Vector3(radius * 1.65, radius * 0.8, radius * 2.35));
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  setTime(seconds) {
    if (!this.mixer) return;
    this.mixer.setTime(Math.min(Math.max(seconds, 0), this.duration));
  }

  render() {
    const width = Math.max(1, this.canvas.clientWidth);
    const height = Math.max(1, this.canvas.clientHeight);
    const pixelRatio = Math.min(devicePixelRatio, 2);
    if (this.canvas.width !== Math.round(width * pixelRatio) || this.canvas.height !== Math.round(height * pixelRatio)) {
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

export class DualMotionViewer {
  constructor(elements) {
    this.elements = elements;
    this.left = new MotionPane(elements.leftCanvas, elements.leftLoading, elements.leftError, elements.leftFps);
    this.right = new MotionPane(elements.rightCanvas, elements.rightLoading, elements.rightError, elements.rightFps);
    this.loadTokens = [0, 0];
    this.time = 0;
    this.duration = 0;
    this.playing = false;
    this.speed = 1;
    this.previous = performance.now();
    elements.play.onclick = () => this.toggle();
    elements.timeline.oninput = () => this.seek(Number(elements.timeline.value) / 1000 * this.duration);
    elements.speed.onchange = () => { this.speed = Number(elements.speed.value); };
    requestAnimationFrame(now => this.animate(now));
  }

  async loadPair(original, augmented) {
    this.pause();
    this.time = 0;
    this.duration = 0;
    this.updateTransport();
    const tokens = this.loadTokens.map((value, index) => {
      this.loadTokens[index] = value + 1;
      return this.loadTokens[index];
    });
    const results = await Promise.allSettled([
      this.left.load(original, () => tokens[0] === this.loadTokens[0]),
      this.right.load(augmented, () => tokens[1] === this.loadTokens[1]),
    ]);
    if (tokens.some((token, index) => token !== this.loadTokens[index])) return;
    this.refreshDuration();
    if (results.every(result => result.status === 'rejected')) throw new Error('Both comparison previews failed to load');
  }

  async loadSlot(slot, asset) {
    const index = Number(slot) === 1 ? 1 : 0;
    const pane = index === 0 ? this.left : this.right;
    const token = ++this.loadTokens[index];
    this.pause();
    const result = await Promise.allSettled([
      pane.load(asset, () => token === this.loadTokens[index]),
    ]);
    if (token !== this.loadTokens[index]) return;
    this.refreshDuration();
    this.seek(Math.min(this.time, this.duration));
    if (result[0].status === 'rejected') throw result[0].reason;
  }

  refreshDuration() {
    this.duration = Math.max(this.left.duration, this.right.duration);
    this.elements.play.disabled = this.duration <= 0;
    this.elements.timeline.disabled = this.duration <= 0;
    this.updateTransport();
  }

  toggle() {
    if (!this.duration) return;
    this.playing = !this.playing;
    this.elements.play.textContent = this.playing ? '❚❚' : '▶';
  }

  pause() {
    this.playing = false;
    this.elements.play.textContent = '▶';
  }

  seek(seconds) {
    this.time = Math.min(Math.max(seconds, 0), this.duration);
    this.left.setTime(this.time);
    this.right.setTime(this.time);
    this.updateTransport();
  }

  updateTransport() {
    const format = value => {
      const minutes = Math.floor(value / 60);
      return `${String(minutes).padStart(2, '0')}:${(value - minutes * 60).toFixed(3).padStart(6, '0')}`;
    };
    this.elements.current.textContent = format(this.time);
    this.elements.duration.textContent = format(this.duration);
    this.elements.timeline.value = this.duration ? String(Math.round(this.time / this.duration * 1000)) : '0';
  }

  setTheme() {
    this.left.setTheme();
    this.right.setTheme();
  }

  animate(now) {
    const dt = Math.min((now - this.previous) / 1000, 0.1);
    this.previous = now;
    if (this.playing && this.duration) {
      this.time += dt * this.speed;
      if (this.time >= this.duration) this.time = 0;
      this.left.setTime(this.time);
      this.right.setTime(this.time);
      this.updateTransport();
    }
    if (!this.elements.leftCanvas.closest('[hidden]')) {
      this.left.render();
      this.right.render();
    }
    requestAnimationFrame(next => this.animate(next));
  }
}
