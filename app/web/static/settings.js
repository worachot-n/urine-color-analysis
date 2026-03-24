/**
 * settings.js — Grid calibration canvas corner-picker + color reference extractor.
 *
 * Grid Calibration flow:
 *   1. User picks an image → drawn on <canvas id="calibCanvas">
 *   2. User clicks 4 corners in order: TL, TR, BR, BL
 *   3. Each click draws a numbered crosshair; corners stored in originalImageCoords
 *   4. "Compute Grid" button POSTs {file, corners} to /settings/grid
 *
 * Color Reference flow:
 *   1. User picks an image → "Extract Colors" POSTs to /settings/colors
 *   2. Server returns swatches (hex + Lab per level)
 *   3. Swatches rendered; "Save Color Reference" is a no-op (server already saved)
 */

'use strict';

// ─── Grid calibration ────────────────────────────────────────────────────────

const gridImageInput  = document.getElementById('gridImageInput');
const canvasWrapper   = document.getElementById('canvasWrapper');
const calibCanvas     = document.getElementById('calibCanvas');
const cornerCount     = document.getElementById('cornerCount');
const resetCornersBtn = document.getElementById('resetCornersBtn');
const computeGridBtn  = document.getElementById('computeGridBtn');
const gridResult      = document.getElementById('gridResult');

const ctx = calibCanvas.getContext('2d');

let originalImage    = null;   // HTMLImageElement
let displayScale     = 1;      // canvas px / original px
let corners          = [];     // [[x,y], ...] in original image coordinates (up to 4)

const CORNER_LABELS  = ['TL', 'TR', 'BR', 'BL'];
const CORNER_COLORS  = ['#0d6efd', '#198754', '#ffc107', '#dc3545'];

// ── Load image onto canvas ────────────────────────────────────────────────────

gridImageInput.addEventListener('change', function () {
    const file = this.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
            originalImage = img;
            corners = [];
            updateCornerCount();
            computeGridBtn.disabled = true;

            // Fit canvas to container width (max 1200px)
            const maxW = Math.min(canvasWrapper.clientWidth || 900, 1200);
            displayScale = Math.min(1, maxW / img.width);
            calibCanvas.width  = Math.round(img.width  * displayScale);
            calibCanvas.height = Math.round(img.height * displayScale);

            canvasWrapper.classList.remove('d-none');
            redraw();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
});

// ── Canvas click → record corner ─────────────────────────────────────────────

calibCanvas.addEventListener('click', function (e) {
    if (!originalImage || corners.length >= 4) return;

    const rect = calibCanvas.getBoundingClientRect();
    // Scale from CSS pixels to canvas pixels, then to original image pixels
    const cssScaleX = calibCanvas.width  / rect.width;
    const cssScaleY = calibCanvas.height / rect.height;
    const canvasX = (e.clientX - rect.left)  * cssScaleX;
    const canvasY = (e.clientY - rect.top)   * cssScaleY;

    const origX = canvasX / displayScale;
    const origY = canvasY / displayScale;

    corners.push([origX, origY]);
    updateCornerCount();

    if (corners.length === 4) {
        computeGridBtn.disabled = false;
    }

    redraw();
});

// ── Reset corners ─────────────────────────────────────────────────────────────

resetCornersBtn.addEventListener('click', function () {
    corners = [];
    updateCornerCount();
    computeGridBtn.disabled = true;
    gridResult.className = 'alert d-none';
    redraw();
});

// ── Redraw canvas ─────────────────────────────────────────────────────────────

function redraw() {
    if (!originalImage) return;
    ctx.clearRect(0, 0, calibCanvas.width, calibCanvas.height);
    ctx.drawImage(originalImage, 0, 0, calibCanvas.width, calibCanvas.height);

    corners.forEach(function ([ox, oy], i) {
        const cx = ox * displayScale;
        const cy = oy * displayScale;
        const color = CORNER_COLORS[i];
        const label = CORNER_LABELS[i];

        // Crosshair lines
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2;
        ctx.beginPath();
        ctx.moveTo(cx - 16, cy); ctx.lineTo(cx + 16, cy);
        ctx.moveTo(cx, cy - 16); ctx.lineTo(cx, cy + 16);
        ctx.stroke();

        // Filled circle
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
        ctx.fill();

        // Label
        ctx.fillStyle   = '#fff';
        ctx.font        = 'bold 12px sans-serif';
        ctx.textAlign   = 'left';
        ctx.fillText(label, cx + 10, cy - 6);
    });

    // Draw polygon outline when all 4 corners are placed
    if (corners.length === 4) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth   = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        corners.forEach(function ([ox, oy], i) {
            const cx = ox * displayScale;
            const cy = oy * displayScale;
            if (i === 0) ctx.moveTo(cx, cy);
            else         ctx.lineTo(cx, cy);
        });
        ctx.closePath();
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

function updateCornerCount() {
    cornerCount.textContent = `${corners.length} / 4`;
}

// ── Submit grid calibration ───────────────────────────────────────────────────

computeGridBtn.addEventListener('click', async function () {
    if (!gridImageInput.files[0] || corners.length !== 4) return;

    computeGridBtn.disabled = true;
    computeGridBtn.textContent = 'Computing…';
    gridResult.className = 'alert d-none';

    const formData = new FormData();
    formData.append('file',    gridImageInput.files[0]);
    formData.append('corners', JSON.stringify(corners));

    try {
        const resp = await fetch('/settings/grid', { method: 'POST', body: formData });
        const data = await resp.json();

        if (resp.ok) {
            gridResult.className   = 'alert alert-success';
            gridResult.textContent = `Grid calibration saved — date: ${data.calibration_date}`;
        } else {
            gridResult.className   = 'alert alert-danger';
            gridResult.textContent = `Error: ${data.detail || 'unknown error'}`;
        }
    } catch (err) {
        gridResult.className   = 'alert alert-danger';
        gridResult.textContent = `Network error: ${err.message}`;
    } finally {
        computeGridBtn.disabled = false;
        computeGridBtn.textContent = 'Compute Grid';
    }
});


// ─── Color reference ─────────────────────────────────────────────────────────

const colorImageInput  = document.getElementById('colorImageInput');
const extractColorsBtn = document.getElementById('extractColorsBtn');
const swatchContainer  = document.getElementById('swatchContainer');
const swatchRow        = document.getElementById('swatchRow');
const colorResult      = document.getElementById('colorResult');
const saveColorsBtn    = document.getElementById('saveColorsBtn');

colorImageInput.addEventListener('change', function () {
    extractColorsBtn.disabled = !this.files[0];
    swatchContainer.classList.add('d-none');
    saveColorsBtn.classList.add('d-none');
    colorResult.className = 'alert d-none';
});

extractColorsBtn.addEventListener('click', async function () {
    if (!colorImageInput.files[0]) return;

    extractColorsBtn.disabled = true;
    extractColorsBtn.textContent = 'Extracting…';
    swatchContainer.classList.add('d-none');
    saveColorsBtn.classList.add('d-none');
    colorResult.className = 'alert d-none';

    const formData = new FormData();
    formData.append('file', colorImageInput.files[0]);

    try {
        const resp = await fetch('/settings/colors', { method: 'POST', body: formData });
        const data = await resp.json();

        if (resp.ok) {
            renderSwatches(data.swatches);
            swatchContainer.classList.remove('d-none');
            colorResult.className   = 'alert alert-success';
            colorResult.textContent = `Colors extracted and saved — date: ${data.calibration_date}`;
            // color.json is already saved server-side; show info-only button
            saveColorsBtn.classList.remove('d-none');
            saveColorsBtn.disabled = true;
            saveColorsBtn.textContent = 'Saved ✓';
        } else {
            colorResult.className   = 'alert alert-danger';
            colorResult.textContent = `Error: ${data.detail || 'unknown error'}`;
        }
    } catch (err) {
        colorResult.className   = 'alert alert-danger';
        colorResult.textContent = `Network error: ${err.message}`;
    } finally {
        extractColorsBtn.disabled = false;
        extractColorsBtn.textContent = 'Extract Colors';
    }
});

function renderSwatches(swatches) {
    swatchRow.innerHTML = '';
    swatches.forEach(function (s) {
        const div = document.createElement('div');
        div.className = 'text-center';
        div.innerHTML = `
            <div class="swatch-box mx-auto mb-1" style="background:${s.hex}"></div>
            <div class="small fw-semibold">L${s.level}</div>
            <div class="small text-muted">${s.hex}</div>
        `;
        swatchRow.appendChild(div);
    });
}
