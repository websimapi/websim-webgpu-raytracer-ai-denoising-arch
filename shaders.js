export const RAYTRACER_WGSL = `
struct Camera {
    position: vec3f,
    forward: vec3f,
    right: vec3f,
    up: vec3f
};

struct Uniforms {
    resolution: vec2f,
    time: f32,
    frame: u32,
    camera_pos: vec3f,
    camera_yaw: f32,
    camera_pitch: f32,
    accumulate: u32, // 0 or 1
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;
// We might use a history buffer for temporal accumulation later
@group(0) @binding(2) var historyTex: texture_2d<f32>; 

// Random Utilities
fn hash(p: u32) -> u32 {
    var x = p;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

var<private> rnd_state: u32;

fn rand() -> f32 {
    rnd_state = hash(rnd_state);
    return f32(rnd_state) / 4294967295.0;
}

fn rand_vec3() -> vec3f {
    return vec3f(rand(), rand(), rand()) * 2.0 - 1.0;
}

fn rand_in_unit_sphere() -> vec3f {
    for(var i = 0; i < 3; i++) {
        let p = rand_vec3();
        if (dot(p, p) < 1.0) { return p; }
    }
    return normalize(rand_vec3());
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

struct Hit {
    hit: bool,
    t: f32,
    pos: vec3f,
    normal: vec3f,
    mat_type: u32, // 0: Diffuse, 1: Metal, 2: Emissive
    albedo: vec3f,
    roughness: f32
};

// Scene Definition
fn sphere_intersect(ray: Ray, center: vec3f, radius: f32, t_min: f32, t_max: f32) -> f32 {
    let oc = ray.origin - center;
    let a = dot(ray.direction, ray.direction);
    let b = dot(oc, ray.direction);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = b * b - a * c;
    
    if (discriminant > 0.0) {
        let temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) { return temp; }
        let temp2 = (-b + sqrt(discriminant)) / a;
        if (temp2 < t_max && temp2 > t_min) { return temp2; }
    }
    return -1.0;
}

fn world_hit(r: Ray) -> Hit {
    var closest: Hit;
    closest.hit = false;
    closest.t = 1e30;

    // Floor
    let t_plane = -(r.origin.y + 1.0) / r.direction.y;
    if (t_plane > 0.001 && t_plane < closest.t) {
        closest.hit = true;
        closest.t = t_plane;
        closest.pos = r.origin + r.direction * t_plane;
        closest.normal = vec3f(0.0, 1.0, 0.0);
        closest.mat_type = 0u;
        
        // Checkerboard pattern
        let scale = 1.0;
        let pattern = (i32(floor(closest.pos.x * scale)) + i32(floor(closest.pos.z * scale))) % 2;
        closest.albedo = select(vec3f(0.1), vec3f(0.8), pattern == 0);
        closest.roughness = 0.8;
    }

    // Sphere 1 (Red)
    let s1_c = vec3f(0.0, 0.0, -1.0);
    let t1 = sphere_intersect(r, s1_c, 0.5, 0.001, closest.t);
    if (t1 > 0.0) {
        closest.hit = true;
        closest.t = t1;
        closest.pos = r.origin + r.direction * t1;
        closest.normal = (closest.pos - s1_c) / 0.5;
        closest.mat_type = 0u;
        closest.albedo = vec3f(0.8, 0.2, 0.2);
        closest.roughness = 0.5;
    }

    // Sphere 2 (Mirror)
    let s2_c = vec3f(1.2, 0.0, -1.5);
    let t2 = sphere_intersect(r, s2_c, 0.5, 0.001, closest.t);
    if (t2 > 0.0) {
        closest.hit = true;
        closest.t = t2;
        closest.pos = r.origin + r.direction * t2;
        closest.normal = (closest.pos - s2_c) / 0.5;
        closest.mat_type = 1u;
        closest.albedo = vec3f(0.9, 0.9, 0.9);
        closest.roughness = 0.0;
    }

    // Sphere 3 (Glass/Blueish)
    let s3_c = vec3f(-1.2, 0.0, -1.5);
    let t3 = sphere_intersect(r, s3_c, 0.5, 0.001, closest.t);
    if (t3 > 0.0) {
        closest.hit = true;
        closest.t = t3;
        closest.pos = r.origin + r.direction * t3;
        closest.normal = (closest.pos - s3_c) / 0.5;
        closest.mat_type = 1u;
        closest.albedo = vec3f(0.2, 0.8, 0.9);
        closest.roughness = 0.1;
    }
    
    // Light Source
    let sL_c = vec3f(0.0, 2.0, -1.0);
    let tL = sphere_intersect(r, sL_c, 0.5, 0.001, closest.t);
    if (tL > 0.0) {
        closest.hit = true;
        closest.t = tL;
        closest.pos = r.origin + r.direction * tL;
        closest.normal = (closest.pos - sL_c) / 0.5;
        closest.mat_type = 2u;
        closest.albedo = vec3f(4.0, 4.0, 3.5); // Very bright
    }

    return closest;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = u32(uniforms.resolution.x);
    let height = u32(uniforms.resolution.y);

    if (id.x >= width || id.y >= height) { return; }

    // Initialize RNG
    let pixel_index = id.y * width + id.x;
    rnd_state = hash(pixel_index ^ hash(uniforms.frame));

    // Camera setup
    let uv = (vec2f(f32(id.x), f32(id.y)) + vec2f(rand(), rand())) / uniforms.resolution * 2.0 - 1.0;
    // Fix aspect ratio
    let aspect = uniforms.resolution.x / uniforms.resolution.y;
    let screen_pos = vec3f(uv.x * aspect, uv.y, -1.0); // -1.0 is focal length

    // Transform ray by camera matrix
    let cz = normalize(uniforms.camera_pos - (uniforms.camera_pos + vec3f(sin(uniforms.camera_yaw), 0.0, cos(uniforms.camera_yaw))));
    // Simple look-from, look-at logic replacement with yaw/pitch for full control:
    // Actually, let's just use the uniforms passed by our Camera class
    // Reconstructing View Matrix from angles:
    let cy = cos(uniforms.camera_yaw);
    let sy = sin(uniforms.camera_yaw);
    let cp = cos(uniforms.camera_pitch);
    let sp = sin(uniforms.camera_pitch);
    
    let fwd = vec3f(sy * cp, sp, -cy * cp); // Forward
    let right = normalize(cross(fwd, vec3f(0.0, 1.0, 0.0)));
    let up = cross(right, fwd);
    
    // Ray dir
    let ray_dir = normalize(fwd * 1.5 + right * uv.x * aspect - up * uv.y); 
    
    var ray = Ray(uniforms.camera_pos, ray_dir);
    var col = vec3f(0.0);
    var throughput = vec3f(1.0);
    
    // Bounces
    for(var i = 0; i < 4; i++) {
        let hit = world_hit(ray);
        
        if (hit.hit) {
            if (hit.mat_type == 2u) {
                // Emissive
                col += throughput * hit.albedo;
                break;
            }
            
            // Scatter
            var scatter_target = vec3f(0.0);
            if (hit.mat_type == 1u) {
                // Metal / Specular
                let reflected = reflect(ray.direction, hit.normal);
                scatter_target = reflected + rand_in_unit_sphere() * hit.roughness;
                throughput *= hit.albedo;
            } else {
                // Diffuse
                scatter_target = hit.normal + rand_in_unit_sphere();
                // Lambertian
                throughput *= hit.albedo * 0.5; 
            }
            
            ray.origin = hit.pos;
            ray.direction = normalize(scatter_target);
        } else {
            // Sky
            let t = 0.5 * (ray.direction.y + 1.0);
            let sky = (1.0 - t) * vec3f(1.0, 1.0, 1.0) + t * vec3f(0.5, 0.7, 1.0);
            col += throughput * sky * 0.5; // Dim sky
            break;
        }
    }
    
    // Accumulation logic could go here, but we output raw noisy frame for denoiser
    var final_col = vec4f(col, 1.0);
    
    if (uniforms.accumulate == 1u && uniforms.frame > 0u) {
        let old = textureLoad(historyTex, vec2<i32>(id.xy), 0);
        let weight = 1.0 / (f32(uniforms.frame) % 100.0 + 1.0); // Basic rolling accumulation reset
        final_col = mix(old, final_col, weight);
    }

    textureStore(outputTex, vec2<i32>(id.xy), final_col);
}
`;

export const DENOISER_WGSL = `
struct Uniforms {
    resolution: vec2f,
    enabled: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba8unorm, write>;

// Gaussian weights for spatial kernel
// Using a 5x5 kernel approximation
fn get_weight(offset: vec2i) -> f32 {
    let d = dot(vec2f(offset), vec2f(offset));
    return exp(-d * 0.1); 
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = u32(uniforms.resolution.x);
    let height = u32(uniforms.resolution.y);

    if (id.x >= width || id.y >= height) { return; }
    
    let coords = vec2<i32>(id.xy);
    let center_color = textureLoad(inputTex, coords, 0).rgb;

    // Pass-through if disabled
    if (uniforms.enabled == 0u) {
        textureStore(outputTex, coords, vec4f(center_color, 1.0));
        return;
    }

    // "AI" Simulation: Bilateral Filter
    // In a real WebNN pipeline, this would be an ONNX model inference call.
    // Here we simulate the edge-preserving smoothing capabilities of such a network.
    
    var sum_col = vec3f(0.0);
    var sum_weight = 0.0;
    
    // 9x9 Kernel for aggressive denoising (simulating larger receptive field)
    let k_size = 3; 
    
    for(var y = -k_size; y <= k_size; y++) {
        for(var x = -k_size; x <= k_size; x++) {
            let offset = vec2<i32>(x, y);
            let sample_pos = coords + offset;
            
            // Bounds check
            if (sample_pos.x < 0 || sample_pos.y < 0 || u32(sample_pos.x) >= width || u32(sample_pos.y) >= height) {
                continue;
            }
            
            let sample_col = textureLoad(inputTex, sample_pos, 0).rgb;
            
            // Spatial weight (Gaussian)
            let offset_f = vec2f(f32(x), f32(y));
            let w_spatial = exp(-dot(offset_f, offset_f) * 0.15);
            
            // Color weight (Edge preservation - similar to features learned by CNN)
            let color_dist = length(center_color - sample_col);
            let w_color = exp(-color_dist * color_dist * 10.0); // High sensitivity to edges
            
            let weight = w_spatial * w_color;
            
            sum_col += sample_col * weight;
            sum_weight += weight;
        }
    }
    
    var final_col = sum_col / sum_weight;
    
    // Simple Tone Mapping
    final_col = final_col / (final_col + vec3f(1.0));
    final_col = pow(final_col, vec3f(1.0/2.2)); // Gamma correction

    textureStore(outputTex, coords, vec4f(final_col, 1.0));
}
`;

