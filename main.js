import { WebGPURenderer } from './renderer.js';
import { Camera } from './camera.js';

const canvas = document.getElementById('gpu-canvas');
const fpsEl = document.getElementById('fps-counter');
const denoiseToggle = document.getElementById('denoise-toggle');
const accumulateToggle = document.getElementById('accumulate-toggle');

const renderer = new WebGPURenderer(canvas);
const camera = new Camera(canvas);

// FPS Tracking
let lastTime = 0;
let frameCount = 0;
let lastFpsTime = 0;

async function init() {
    try {
        await renderer.init();
        requestAnimationFrame(loop);
    } catch (err) {
        document.body.innerHTML = `<div style="color:white; padding:20px;">
            <h1>Error</h1>
            <p>${err.message}</p>
            <p>Please use a browser that supports WebGPU (Chrome 113+ or Edge).</p>
        </div>`;
        console.error(err);
    }
}

function loop(timestamp) {
    const dt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;

    // Logic
    camera.update(dt);
    
    // Check if camera moved significantly to reset accumulation
    // (Simplification: we just reset every frame unless Accumulate toggle is ON and no input)
    // For this demo, we handle accumulation reset in renderer based on toggle
    
    // UI Updates
    if (renderer.denoiseEnabled !== denoiseToggle.checked) {
        renderer.denoiseEnabled = denoiseToggle.checked;
    }
    
    if (renderer.accumulate !== accumulateToggle.checked) {
        renderer.accumulate = accumulateToggle.checked;
        renderer.frame = 0;
    }

    // Render
    renderer.render(camera);

    // FPS
    frameCount++;
    if (timestamp - lastFpsTime >= 1000) {
        fpsEl.innerText = frameCount;
        frameCount = 0;
        lastFpsTime = timestamp;
    }

    requestAnimationFrame(loop);
}

init();

