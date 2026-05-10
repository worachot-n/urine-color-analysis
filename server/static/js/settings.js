/**
 * settings.js — interactive slot assignment grid.
 *
 * State: { rows, cols, cells: { [cellIdx]: {slot_id, is_reference, ref_level,
 *                                            is_white_reference} } }
 * All mutations update state then re-render the affected cell(s).
 * GET /api/slots loads saved config; POST /api/slots persists it.
 *
 * Cell types are mutually exclusive: a cell is either a colour reference
 * (is_reference + ref_level), a sample, or a white reference (neutral patch
 * for white-balance correction).
 */

const REF_BG = ['#fce375','#499d71','#4a8c54','#1b593c','#f05152'];
const REF_FG = ['#122017','#122017','#122017','#ffffff','#ffffff'];
const WB_BG  = '#bfdbfe';
const WB_FG  = '#1e3a8a';

let state = { rows: 13, cols: 15, cells: {} };
let activeCellIdx = null;

// ---------------------------------------------------------------------------
// Grid init
// ---------------------------------------------------------------------------
function initGrid(rows, cols) {
  state.rows = rows;
  state.cols = cols;

  // Header row
  const thead = document.getElementById('grid-thead');
  thead.innerHTML = '';
  const hRow = document.createElement('tr');
  const thEmpty = document.createElement('th');
  thEmpty.className = 'px-2 py-1 text-gray-500 text-xs';
  hRow.appendChild(thEmpty);
  for (let c = 1; c <= cols; c++) {
    const th = document.createElement('th');
    th.className = 'px-1 py-1 text-gray-500 text-xs font-normal';
    th.textContent = c;
    hRow.appendChild(th);
  }
  thead.appendChild(hRow);

  // Body
  const tbody = document.getElementById('grid-tbody');
  tbody.innerHTML = '';
  for (let r = 1; r <= rows; r++) {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.className = 'px-2 py-1 text-gray-500 text-xs font-normal text-right';
    th.textContent = r;
    tr.appendChild(th);
    for (let c = 1; c <= cols; c++) {
      const idx = (r - 1) * cols + c;
      const td = document.createElement('td');
      td.id = `cell-${idx}`;
      td.className = 'grid-cell';
      td.dataset.idx = idx;
      td.addEventListener('click', () => openPopup(idx));
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  renderAllCells();
}

// ---------------------------------------------------------------------------
// Cell rendering
// ---------------------------------------------------------------------------
function renderCell(idx) {
  const td = document.getElementById(`cell-${idx}`);
  if (!td) return;
  const cell = state.cells[idx];
  if (!cell) {
    td.style.background = '#234228';
    td.style.color = '#9ca3af';
    td.textContent = '';
    return;
  }
  if (cell.is_reference && cell.ref_level != null) {
    const lvl = parseInt(cell.ref_level);
    td.style.background = REF_BG[lvl] || '#888';
    td.style.color = REF_FG[lvl] || '#000';
    td.textContent = `ขวดอ้างอิง สี ${lvl}`;
  } else if (cell.is_white_reference) {
    td.style.background = WB_BG;
    td.style.color = WB_FG;
    td.textContent = cell.slot_id ? `WB ${cell.slot_id}` : 'WB';
  } else {
    td.style.background = '#f9fafb';
    td.style.color = '#111827';
    td.textContent = cell.slot_id;
  }
}

function renderAllCells() {
  for (let r = 1; r <= state.rows; r++) {
    for (let c = 1; c <= state.cols; c++) {
      renderCell((r - 1) * state.cols + c);
    }
  }
}

// ---------------------------------------------------------------------------
// Popup
// ---------------------------------------------------------------------------
function openPopup(idx) {
  activeCellIdx = idx;
  const r = Math.floor((idx - 1) / state.cols) + 1;
  const c = (idx - 1) % state.cols + 1;

  document.getElementById('popup-title').textContent = `แถว ${r}, คอลัมน์ ${c}  (เซลล์ ${idx})`;
  const cell = state.cells[idx] || {
    slot_id: '', is_reference: false, ref_level: null, is_white_reference: false,
  };
  document.getElementById('popup-slot-id').value = cell.slot_id || '';
  const isRef = Boolean(cell.is_reference);
  const isWb  = Boolean(cell.is_white_reference);
  document.getElementById('popup-is-ref').checked = isRef;
  document.getElementById('popup-is-wb').checked  = isWb;
  document.getElementById('popup-level').value = cell.ref_level ?? 0;
  document.getElementById('popup-level-row').classList.toggle('hidden', !isRef);

  const popup = document.getElementById('popup');
  popup.style.display = 'block';

  // Position near the clicked cell
  const td = document.getElementById(`cell-${idx}`);
  if (td) {
    const rect = td.getBoundingClientRect();
    let left = rect.right + 8 + window.scrollX;
    let top  = rect.top  + window.scrollY;
    // keep popup in viewport
    if (left + 240 > window.innerWidth) left = rect.left - 248 + window.scrollX;
    popup.style.left = `${Math.max(4, left)}px`;
    popup.style.top  = `${Math.max(4, top)}px`;
  }
}

// Mutually exclusive: ticking one auto-unticks the other.
document.getElementById('popup-is-ref').addEventListener('change', function() {
  document.getElementById('popup-level-row').classList.toggle('hidden', !this.checked);
  if (this.checked) document.getElementById('popup-is-wb').checked = false;
});
document.getElementById('popup-is-wb').addEventListener('change', function() {
  if (this.checked) {
    document.getElementById('popup-is-ref').checked = false;
    document.getElementById('popup-level-row').classList.add('hidden');
  }
});

function popupSave() {
  const slotId = document.getElementById('popup-slot-id').value.trim();
  if (!slotId) { showToast('กรุณากรอกรหัสช่อง', 'error'); return; }
  const isRef = document.getElementById('popup-is-ref').checked;
  const isWb  = document.getElementById('popup-is-wb').checked;
  if (isRef && isWb) { showToast('เซลล์ไม่สามารถเป็นทั้งสองประเภทพร้อมกัน', 'error'); return; }
  const refLevel = isRef ? parseInt(document.getElementById('popup-level').value) : null;
  state.cells[activeCellIdx] = {
    slot_id: slotId,
    is_reference: isRef,
    ref_level: refLevel,
    is_white_reference: isWb,
  };
  renderCell(activeCellIdx);
  closePopup();
}

function popupClear() {
  delete state.cells[activeCellIdx];
  renderCell(activeCellIdx);
  closePopup();
}

function closePopup() {
  document.getElementById('popup').style.display = 'none';
  activeCellIdx = null;
}

// Close popup on outside click
document.addEventListener('click', e => {
  const popup = document.getElementById('popup');
  if (popup.style.display !== 'none' && !popup.contains(e.target)) {
    const td = e.target.closest('[data-idx]');
    if (!td) closePopup();
  }
});

// ---------------------------------------------------------------------------
// Config load / save / reset
// ---------------------------------------------------------------------------
async function loadConfig() {
  try {
    const res = await fetch('/api/slots');
    const data = await res.json();
    state.rows = data.rows || 13;
    state.cols = data.cols || 15;
    // Convert string keys to int
    state.cells = {};
    for (const [k, v] of Object.entries(data.cells || {})) {
      state.cells[parseInt(k)] = v;
    }
    renderAllCells();
  } catch (e) {
    showToast('โหลดการตั้งค่าล้มเหลว: ' + e.message, 'error');
  }
}

function resetAll() {
  if (!confirm('ล้างการกำหนดช่องทั้งหมด? (ยังไม่บันทึกจนกว่าจะกดบันทึก)')) return;
  state.cells = {};
  renderAllCells();
}

async function saveConfig() {
  try {
    const body = {
      rows: state.rows,
      cols: state.cols,
      cells: {},
    };
    for (const [k, v] of Object.entries(state.cells)) {
      body.cells[k] = v;
    }
    const res = await fetch('/api/slots', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).detail || 'Server error');
    const data = await res.json();
    showToast(`บันทึกแล้ว — ${data.cells} เซลล์`, 'success');
  } catch (e) {
    showToast('บันทึกล้มเหลว: ' + e.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------
function showToast(msg, type = 'info') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'show ' + (type === 'error' ? 'bg-red-700 text-white' : 'bg-green-700 text-white');
  el.style.display = 'block';
  el.style.opacity = '1';
  setTimeout(() => {
    el.style.opacity = '0';
    setTimeout(() => { el.style.display = 'none'; }, 300);
  }, 3000);
}
