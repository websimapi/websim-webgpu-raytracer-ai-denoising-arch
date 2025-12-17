import { RAYTRACER_WGSL, DENOISER_WGSL } from './shaders.js';

export class WebGPURenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.adapter = null;
        this.device = null;
        this.context = null;
        this.width = 0;
        this.height = 0;
        this.frame = 0;
        this.accumulate = false;
        this.denoiseEnabled = true;
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error("WebGPU not supported on this browser.");
        }

        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) {
            throw new Error("No WebGPU adapter found.");
        }

        this.device = await this.adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu');
        
        // Initial resize to ensure textures exist before pipeline setup
        const initialWidth = Math.max(1, this.canvas.clientWidth);
        const initialHeight = Math.max(1, this.canvas.clientHeight);
        this.resize(initialWidth, initialHeight);

        // Resize observer
        const observer = new ResizeObserver(entries => {
            for (let entry of entries) {
                const width = entry.contentBoxSize[0].inlineSize;
                const height = entry.contentBoxSize[0].blockSize;
                this.resize(width, height);
            }
        });
        observer.observe(this.canvas);
        
        await this.setupPipelines();
    }
    
    resize(width, height) {
        if (width === 0 || height === 0) return;
        
        // Clamp for performance on high DPI, ensure min 1
        this.width = Math.max(1, Math.min(width, 1920)); 
        this.height = Math.max(1, Math.min(height, 1080));
        
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        
        // Use rgba8unorm for compatibility with storage textures (compute write)
        this.context.configure({
            device: this.device,
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
            alphaMode: 'premultiplied'
        });

        this.createTextures();
        // Reset history accumulation
        this.frame = 0;
    }

    createTextures() {
        if (this.traceTexture) this.traceTexture.destroy();
        if (this.historyTexture) this.historyTexture.destroy();

        // 1. Raytracer Output (HDR, float)
        this.traceTexture = this.device.createTexture({
            size: [this.width, this.height, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
        });

        // 2. History Texture (For temporal accumulation)
        this.historyTexture = this.device.createTexture({
            size: [this.width, this.height, 1],
            format: 'rgba16float', // Needs to match trace for copy/loading
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING
        });

        // Re-create bind groups since texture views changed
        if (this.raytracePipeline) this.createBindGroups();
    }

    async setupPipelines() {
        // --- Raytracer Pipeline ---
        this.raytraceModule = this.device.createShaderModule({ code: RAYTRACER_WGSL });
        this.raytracePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.raytraceModule, entryPoint: 'main' }
        });

        // --- Denoiser Pipeline ---
        this.denoiseModule = this.device.createShaderModule({ code: DENOISER_WGSL });
        this.denoisePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.denoiseModule, entryPoint: 'main' }
        });

        // Uniform Buffer
        this.uniformBufferSize = 32 + 16; // vec2 + f32 + u32 + vec3 + f32 + f32 + u32 + padding
        this.uniformBuffer = this.device.createBuffer({
            size: this.uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.denoiseUniformBuffer = this.device.createBuffer({
            size: 16, // vec2 + u32 + padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.createBindGroups();
    }

    createBindGroups() {
        if (!this.traceTexture || !this.historyTexture) return;

        this.raytraceBindGroup = this.device.createBindGroup({
            layout: this.raytracePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: this.traceTexture.createView() },
                { binding: 2, resource: this.historyTexture.createView() } // Actually used as read-only in shader, but we might ping-pong later
            ]
        });
    }

    render(camera) {
        if (!this.device || this.width === 0) return;

        // 1. Update Uniforms
        const uniforms = new ArrayBuffer(this.uniformBufferSize);
        const f32 = new Float32Array(uniforms);
        const u32 = new Uint32Array(uniforms);

        f32[0] = this.width;
        f32[1] = this.height;
        f32[2] = performance.now() / 1000.0;
        u32[3] = this.frame;
        
        f32[4] = camera.position.x;
        f32[5] = camera.position.y;
        f32[6] = camera.position.z;
        f32[7] = 0; // padding
        
        f32[8] = camera.yaw;
        f32[9] = camera.pitch;
        u32[10] = this.accumulate ? 1 : 0;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

        // Denoiser Uniforms
        const denoiseUniforms = new Float32Array([this.width, this.height]);
        this.device.queue.writeBuffer(this.denoiseUniformBuffer, 0, denoiseUniforms);
        this.device.queue.writeBuffer(this.denoiseUniformBuffer, 8, new Uint32Array([this.denoiseEnabled ? 1 : 0]));

        // 2. Command Encoder
        const commandEncoder = this.device.createCommandEncoder();

        // --- Pass 1: Raytrace ---
        if (this.raytraceBindGroup) {
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.raytracePipeline);
            passEncoder.setBindGroup(0, this.raytraceBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
            passEncoder.end();
        }

        // --- Pass 2: Denoise (or copy to screen) ---
        // We output from Raytrace (traceTexture) -> Denoiser -> Screen (Canvas Texture)
        
        const canvasTexture = this.context.getCurrentTexture();
        
        const denoiseBindGroup = this.device.createBindGroup({
            layout: this.denoisePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.denoiseUniformBuffer } },
                { binding: 1, resource: this.traceTexture.createView() },
                { binding: 2, resource: canvasTexture.createView() }
            ]
        });

        const denoisePass = commandEncoder.beginComputePass();
        denoisePass.setPipeline(this.denoisePipeline);
        denoisePass.setBindGroup(0, denoiseBindGroup);
        denoisePass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        denoisePass.end();

        // --- Pass 3: Copy to History (for next frame accumulation) ---
        if (this.accumulate) {
            commandEncoder.copyTextureToTexture(
                { texture: this.traceTexture },
                { texture: this.historyTexture },
                [this.width, this.height, 1]
            );
        }

        this.device.queue.submit([commandEncoder.finish()]);

        if (this.accumulate) {
            this.frame++;
        } else {
            this.frame = 0; // Keep regenerating noise if not accumulating
        }
    }
}

