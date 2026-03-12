// -----------------------------------------------------------------------
// home.js — Bioluminescent particle canvas for home page
// -----------------------------------------------------------------------

const canvas = document.getElementById('ocean-canvas');
const ctx    = canvas.getContext('2d');

let W, H, particles = [];

function resize() {
  W = canvas.width  = window.innerWidth;
  H = canvas.height = window.innerHeight;
}

function spawnParticles(n) {
  for (let i = 0; i < n; i++) {
    particles.push({
      x:       Math.random() * W,
      y:       Math.random() * H,
      r:       Math.random() * 1.8 + 0.4,
      vx:      (Math.random() - 0.5) * 0.25,
      vy:      -(Math.random() * 0.4 + 0.1),
      life:    Math.random(),
      maxLife: Math.random() * 0.6 + 0.4,
      hue:     Math.random() > 0.5 ? 180 : 150,
    });
  }
}

function draw() {
  ctx.clearRect(0, 0, W, H);

  particles.forEach(p => {
    p.life += 0.003;
    if (p.life > p.maxLife) p.life = 0;

    const alpha = Math.sin((p.life / p.maxLife) * Math.PI) * 0.7;

    // Core dot
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = `hsla(${p.hue}, 100%, 70%, ${alpha})`;
    ctx.fill();

    // Soft glow
    const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4);
    grad.addColorStop(0, `hsla(${p.hue}, 100%, 70%, ${alpha * 0.3})`);
    grad.addColorStop(1, 'transparent');
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r * 4, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    p.x += p.vx;
    p.y += p.vy;

    if (p.y < -10) { p.y = H + 10; p.x = Math.random() * W; }
  });

  requestAnimationFrame(draw);
}

resize();
spawnParticles(120);
draw();
window.addEventListener('resize', resize);
