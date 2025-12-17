import nipplejs from 'nipplejs';

export class Camera {
    constructor(element) {
        this.position = { x: 0, y: 1.0, z: 3.5 };
        this.yaw = 0;   // Rotation around Y
        this.pitch = 0; // Rotation around X (clamped)
        
        this.keys = {};
        this.drag = { isDragging: false, lastX: 0, lastY: 0 };
        
        this.setupKeyboard();
        this.setupMouse(element);
        this.setupJoystick();
    }

    setupKeyboard() {
        window.addEventListener('keydown', (e) => this.keys[e.code] = true);
        window.addEventListener('keyup', (e) => this.keys[e.code] = false);
    }

    setupMouse(element) {
        element.addEventListener('mousedown', (e) => {
            this.drag.isDragging = true;
            this.drag.lastX = e.clientX;
            this.drag.lastY = e.clientY;
        });

        window.addEventListener('mouseup', () => this.drag.isDragging = false);
        
        window.addEventListener('mousemove', (e) => {
            if (!this.drag.isDragging) return;
            
            const dx = e.clientX - this.drag.lastX;
            const dy = e.clientY - this.drag.lastY;
            
            this.yaw -= dx * 0.005;
            this.pitch -= dy * 0.005;
            
            // Clamp pitch to prevent flipping
            this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch));
            
            this.drag.lastX = e.clientX;
            this.drag.lastY = e.clientY;
        });
        
        // Mobile touch swipe for look
        element.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) { // Single touch only
                this.drag.lastX = e.touches[0].clientX;
                this.drag.lastY = e.touches[0].clientY;
            }
        });
        
        element.addEventListener('touchmove', (e) => {
            if (e.touches.length === 1) {
                const dx = e.touches[0].clientX - this.drag.lastX;
                const dy = e.touches[0].clientY - this.drag.lastY;
                
                this.yaw -= dx * 0.01;
                this.pitch -= dy * 0.01;
                this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch));
                
                this.drag.lastX = e.touches[0].clientX;
                this.drag.lastY = e.touches[0].clientY;
            }
        });
    }
    
    setupJoystick() {
        const zone = document.getElementById('joystick-zone');
        // Only initialize if touch capable or always for testing
        const options = {
            zone: zone,
            mode: 'static',
            position: { left: '50%', top: '50%' },
            color: 'white'
        };
        
        this.joystick = nipplejs.create(options);
        
        this.joystickInput = { x: 0, y: 0 };
        
        this.joystick.on('move', (evt, data) => {
            if(data.vector) {
                this.joystickInput.x = data.vector.x;
                this.joystickInput.y = -data.vector.y; // Invert Y for forward
            }
        });
        
        this.joystick.on('end', () => {
            this.joystickInput = { x: 0, y: 0 };
        });
    }

    update(dt) {
        const speed = 4.0 * dt;
        
        let moveZ = 0;
        let moveX = 0;
        
        // Keyboard
        if (this.keys['KeyW']) moveZ += 1;
        if (this.keys['KeyS']) moveZ -= 1;
        if (this.keys['KeyA']) moveX -= 1;
        if (this.keys['KeyD']) moveX += 1;
        
        // Joystick
        moveX += this.joystickInput.x;
        moveZ += this.joystickInput.y;
        
        // Normalize if moving diagonally
        const len = Math.sqrt(moveX*moveX + moveZ*moveZ);
        if (len > 0.001) {
             moveX /= len;
             moveZ /= len;
        }

        // Apply rotation to movement
        const sy = Math.sin(this.yaw);
        const cy = Math.cos(this.yaw);
        
        const fwdX = sy;
        const fwdZ = -cy;
        const rightX = cy;
        const rightZ = sy;
        
        this.position.x += (fwdX * moveZ + rightX * moveX) * speed;
        this.position.z += (fwdZ * moveZ + rightZ * moveX) * speed;
        
        if (this.keys['KeyQ']) this.position.y -= speed;
        if (this.keys['KeyE']) this.position.y += speed;
    }
}

