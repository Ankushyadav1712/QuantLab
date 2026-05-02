// Toast notifications + styled confirm dialog.
// Replaces the browser's native `alert()` / `confirm()` (the ones that
// prepend "localhost says…" and look nothing like the rest of the app).

let container = null;

function ensureContainer() {
  // Recreate if the container was detached (e.g. document.body cleared in tests
  // or by a hot-reload).  Without this guard the cached ref keeps pointing at
  // a detached node and toasts vanish silently.
  if (container && container.isConnected) return container;
  container = document.createElement('div');
  container.className = 'toast-container';
  document.body.appendChild(container);
  return container;
}

const ICONS = {
  success: '✓',
  error: '✕',
  warning: '⚠',
  info: 'ℹ',
};

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/**
 * Show a toast in the top-right corner.
 *   toast('Saved successfully', 'success')
 *   toast('Simulation failed: ...', 'error', { duration: 8000 })
 *
 * Returns a function that dismisses the toast early.
 */
export function toast(message, kind = 'info', opts = {}) {
  const { duration = 4500, title = null } = opts;
  const host = ensureContainer();

  const el = document.createElement('div');
  el.className = `toast toast-${kind}`;
  el.setAttribute('role', kind === 'error' || kind === 'warning' ? 'alert' : 'status');
  el.innerHTML = `
    <span class="toast-icon">${ICONS[kind] || ICONS.info}</span>
    <div class="toast-body">
      ${title ? `<div class="toast-title">${escapeHtml(title)}</div>` : ''}
      <div class="toast-msg">${escapeHtml(message)}</div>
    </div>
    <button class="toast-close" type="button" aria-label="Close">×</button>
  `;
  host.appendChild(el);

  // Animate in next frame so the CSS transition fires
  requestAnimationFrame(() => el.classList.add('show'));

  const dismiss = () => {
    if (!el.isConnected) return;
    el.classList.remove('show');
    el.classList.add('hide');
    setTimeout(() => el.remove(), 250);
  };

  const timer = duration > 0 ? setTimeout(dismiss, duration) : null;
  el.querySelector('.toast-close').addEventListener('click', () => {
    if (timer) clearTimeout(timer);
    dismiss();
  });

  return dismiss;
}

/**
 * Promise-based replacement for window.confirm().
 *
 *   const ok = await confirmDialog({
 *     title: 'Delete alpha?',
 *     message: 'This cannot be undone.',
 *     confirmLabel: 'Delete',
 *     danger: true,
 *   });
 *   if (ok) ...
 */
export function confirmDialog({
  title = 'Are you sure?',
  message = '',
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  danger = false,
} = {}) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    const modal = document.createElement('div');
    modal.className = 'modal modal-confirm';
    modal.innerHTML = `
      <h2>${escapeHtml(title)}</h2>
      ${message ? `<p class="confirm-message">${escapeHtml(message)}</p>` : ''}
      <div class="modal-actions">
        <button type="button" data-role="cancel">${escapeHtml(cancelLabel)}</button>
        <button type="button" data-role="confirm" class="${danger ? 'danger' : 'primary'}">${escapeHtml(confirmLabel)}</button>
      </div>
    `;
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    const close = (value) => {
      overlay.remove();
      document.removeEventListener('keydown', onKey);
      resolve(value);
    };
    const onKey = (e) => {
      if (e.key === 'Escape') close(false);
      else if (e.key === 'Enter') close(true);
    };
    document.addEventListener('keydown', onKey);

    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) close(false);
    });
    modal.querySelector('[data-role="cancel"]').addEventListener('click', () => close(false));
    modal.querySelector('[data-role="confirm"]').addEventListener('click', () => close(true));

    // Focus the primary button so Enter / Esc work without a click
    modal.querySelector('[data-role="confirm"]').focus();
  });
}
