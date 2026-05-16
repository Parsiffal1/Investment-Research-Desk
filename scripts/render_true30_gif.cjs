#!/usr/bin/env node
const { chromium } = require('playwright');
const { mkdirSync, rmSync } = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function arg(name, def = null) {
  const found = process.argv.find(a => a.startsWith(`--${name}=`));
  return found ? found.slice(name.length + 3) : def;
}

const htmlFile = process.argv[2];
if (!htmlFile || htmlFile.startsWith('--')) {
  console.error('Usage: node scripts/render_true30_gif.cjs <html-file> [--duration=10.5] [--fps=30] [--width=1920] [--height=1080] [--gif-width=820]');
  process.exit(1);
}

const duration = parseFloat(arg('duration', '10.5'));
const fps = parseInt(arg('fps', '30'), 10);
const width = parseInt(arg('width', '1920'), 10);
const height = parseInt(arg('height', '1080'), 10);
const gifWidth = parseInt(arg('gif-width', '820'), 10);
const htmlAbs = path.resolve(htmlFile);
const dir = path.dirname(htmlAbs);
const base = path.basename(htmlAbs, path.extname(htmlAbs));
const frameDir = path.join(dir, `.${base}-frames`);
const mp4Out = path.join(dir, `${base}.mp4`);
const gifOut = path.join(dir, `${base}.gif`);
const palette = path.join(dir, `.${base}-palette.png`);
const totalFrames = Math.round(duration * fps);

function run(cmd, args) {
  const res = spawnSync(cmd, args, { stdio: 'inherit' });
  if (res.status !== 0) process.exit(res.status ?? 1);
}

(async function main() {
  rmSync(frameDir, { recursive: true, force: true });
  mkdirSync(frameDir, { recursive: true });
  rmSync(mp4Out, { force: true });
  rmSync(gifOut, { force: true });
  rmSync(palette, { force: true });

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width, height } });
  await page.addInitScript(() => { window.__gifExport = true; });
  await page.goto(`file://${htmlAbs}`, { waitUntil: 'load', timeout: 60000 });
  await page.waitForFunction(() => window.__ready === true, { timeout: 10000 });

  for (let i = 0; i < totalFrames; i++) {
    const sec = i / fps;
    await page.evaluate((t) => {
      if (typeof window.__seek === 'function') window.__seek(t);
    }, sec);
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    const file = path.join(frameDir, `frame-${String(i).padStart(4, '0')}.png`);
    await page.screenshot({ path: file });
    if ((i + 1) % 30 === 0 || i === totalFrames - 1) {
      console.log(`captured ${i + 1}/${totalFrames}`);
    }
  }
  await browser.close();

  run('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', path.join(frameDir, 'frame-%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.0', '-crf', '18', '-preset', 'medium', '-movflags', '+faststart', mp4Out]);
  run('ffmpeg', ['-y', '-loglevel', 'error', '-i', mp4Out, '-vf', `fps=${fps},scale=${gifWidth}:-1:flags=lanczos,palettegen=stats_mode=diff:max_colors=128`, palette]);
  run('ffmpeg', ['-y', '-loglevel', 'error', '-i', mp4Out, '-i', palette, '-lavfi', `fps=${fps},scale=${gifWidth}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none`, gifOut]);

  rmSync(palette, { force: true });
  rmSync(frameDir, { recursive: true, force: true });
  console.log(`done: ${mp4Out}`);
  console.log(`done: ${gifOut}`);
})().catch(err => {
  console.error(err);
  process.exit(1);
});
