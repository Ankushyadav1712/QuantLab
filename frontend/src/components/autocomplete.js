// In-editor autocomplete — floating dropdown showing operator + field
// candidates as the user types.  Lives over the existing textarea / overlay
// editor without disturbing the syntax-highlight layer.
//
// Filtering rules (in priority order):
//   1. exact case-insensitive prefix match
//   2. substring match in name
//   3. substring match in description
// Top 8 matches shown.  Arrow keys navigate, Enter/Tab inserts, ESC dismisses.
// Insertion behavior: operators get an auto-paren ("ts_mean" → "ts_mean(") and
// the caret lands inside.  Fields don't.

const MAX_ITEMS = 8;
const TRIGGER_RE = /[A-Za-z_][A-Za-z0-9_]*$/;

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function rankMatches(items, partial) {
  if (!partial) return [];
  const q = partial.toLowerCase();
  const exact = [];
  const sub = [];
  const desc = [];
  for (const item of items) {
    const name = item.name.toLowerCase();
    if (name.startsWith(q)) exact.push(item);
    else if (name.includes(q)) sub.push(item);
    else if ((item.description || '').toLowerCase().includes(q)) desc.push(item);
  }
  // Within each bucket, prefer shorter names (closer matches)
  exact.sort((a, b) => a.name.length - b.name.length);
  sub.sort((a, b) => a.name.length - b.name.length);
  return [...exact, ...sub, ...desc].slice(0, MAX_ITEMS);
}

// Compute a (left, top) px coordinate for the caret using a hidden mirror
// <div> that mimics the textarea's font / padding / size.  Standard trick.
function getCaretCoords(textarea) {
  const mirror = document.createElement('div');
  const style = window.getComputedStyle(textarea);
  // Copy properties that affect text layout
  const props = [
    'boxSizing', 'width', 'height', 'overflowX', 'overflowY',
    'borderTopWidth', 'borderRightWidth', 'borderBottomWidth', 'borderLeftWidth',
    'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
    'fontStyle', 'fontVariant', 'fontWeight', 'fontStretch', 'fontSize',
    'fontSizeAdjust', 'lineHeight', 'fontFamily',
    'textAlign', 'textTransform', 'textIndent', 'textDecoration',
    'letterSpacing', 'wordSpacing', 'tabSize',
    'whiteSpace', 'wordWrap', 'wordBreak',
  ];
  for (const p of props) mirror.style[p] = style[p];
  mirror.style.position = 'absolute';
  mirror.style.visibility = 'hidden';
  mirror.style.top = '0';
  mirror.style.left = '-9999px';
  mirror.style.whiteSpace = 'pre-wrap';
  mirror.style.wordWrap = 'break-word';

  const value = textarea.value.substring(0, textarea.selectionEnd);
  mirror.textContent = value;
  // Inject a marker span at the caret position so we can measure it
  const marker = document.createElement('span');
  marker.textContent = '​';  // zero-width space
  mirror.appendChild(marker);

  document.body.appendChild(mirror);
  const rect = textarea.getBoundingClientRect();
  const markerRect = marker.getBoundingClientRect();
  const mirrorRect = mirror.getBoundingClientRect();
  const left = rect.left + (markerRect.left - mirrorRect.left) - textarea.scrollLeft;
  const top = rect.top + (markerRect.top - mirrorRect.top) - textarea.scrollTop;
  document.body.removeChild(mirror);

  return { left, top, lineHeight: parseFloat(style.lineHeight) || 18 };
}

export function createAutocomplete({ items, textarea }) {
  const dropdown = document.createElement('div');
  dropdown.className = 'autocomplete';
  dropdown.style.display = 'none';
  document.body.appendChild(dropdown);

  let currentMatches = [];
  let selectedIdx = 0;
  let isOpen = false;
  let partialStart = 0;  // textarea index where the partial token starts

  function hide() {
    if (!isOpen) return;
    dropdown.style.display = 'none';
    isOpen = false;
  }

  function showAt(matches, partial, startIdx) {
    if (matches.length === 0) {
      hide();
      return;
    }
    currentMatches = matches;
    selectedIdx = 0;
    partialStart = startIdx;
    render();
    const coords = getCaretCoords(textarea);
    dropdown.style.left = `${Math.round(coords.left)}px`;
    dropdown.style.top = `${Math.round(coords.top + coords.lineHeight + 2)}px`;
    dropdown.style.display = '';
    isOpen = true;
  }

  function render() {
    dropdown.innerHTML = currentMatches.map((it, i) => {
      const cls = i === selectedIdx ? 'autocomplete-item selected' : 'autocomplete-item';
      const kindLabel = it.kind === 'operator' ? 'op' : 'field';
      const argsHtml = it.kind === 'operator' && it.args
        ? `<span class="autocomplete-args">${escapeHtml(it.args)}</span>`
        : '';
      return `
        <div class="${cls}" data-idx="${i}">
          <span class="autocomplete-kind autocomplete-kind-${it.kind}">${kindLabel}</span>
          <span class="autocomplete-name">${escapeHtml(it.name)}</span>
          ${argsHtml}
          <span class="autocomplete-cat">${escapeHtml(it.category || '')}</span>
          <div class="autocomplete-desc">${escapeHtml(it.description || '')}</div>
        </div>
      `;
    }).join('');
    // Wire click-to-insert
    dropdown.querySelectorAll('.autocomplete-item').forEach((el) => {
      el.addEventListener('mousedown', (e) => {
        // mousedown (not click) so the textarea doesn't lose focus before insertion
        e.preventDefault();
        const idx = Number(el.dataset.idx);
        insert(currentMatches[idx]);
      });
    });
  }

  function insert(item) {
    const value = textarea.value;
    const caret = textarea.selectionEnd;
    const before = value.slice(0, partialStart);
    const after = value.slice(caret);
    let inserted = item.name;
    let cursorOffset = inserted.length;
    if (item.kind === 'operator') {
      // Auto-paren: place caret inside the parens
      inserted += '(';
      cursorOffset = inserted.length;
    }
    textarea.value = before + inserted + after;
    const newCaret = before.length + cursorOffset;
    textarea.setSelectionRange(newCaret, newCaret);
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    textarea.focus();
    hide();
  }

  function maybeOpen() {
    const value = textarea.value;
    const caret = textarea.selectionEnd;
    // Don't trigger if there's a selection range
    if (textarea.selectionStart !== caret) {
      hide();
      return;
    }
    const before = value.slice(0, caret);
    const m = before.match(TRIGGER_RE);
    if (!m) {
      hide();
      return;
    }
    const partial = m[0];
    const matches = rankMatches(items, partial);
    if (matches.length === 0) {
      hide();
      return;
    }
    showAt(matches, partial, before.length - partial.length);
  }

  // ---------- Wire into textarea ----------

  textarea.addEventListener('input', () => {
    // Defer until after the input event has updated the value
    requestAnimationFrame(maybeOpen);
  });
  textarea.addEventListener('keydown', (e) => {
    if (!isOpen) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIdx = (selectedIdx + 1) % currentMatches.length;
      render();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIdx = (selectedIdx - 1 + currentMatches.length) % currentMatches.length;
      render();
    } else if (e.key === 'Enter' || e.key === 'Tab') {
      e.preventDefault();
      insert(currentMatches[selectedIdx]);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      hide();
    }
  });
  textarea.addEventListener('blur', () => {
    // Small delay so click on dropdown items still fires
    setTimeout(hide, 100);
  });
  // Hide on click outside the dropdown
  document.addEventListener('mousedown', (e) => {
    if (!dropdown.contains(e.target) && e.target !== textarea) hide();
  });

  return {
    isOpen: () => isOpen,
    hide,
  };
}
