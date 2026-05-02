import { afterEach, describe, expect, it, vi } from 'vitest';
import { confirmDialog, toast } from '../src/ui/toast.js';

describe('toast()', () => {
  afterEach(() => {
    document.body.innerHTML = '';
    vi.useRealTimers();
  });

  it('appends a toast element with the right kind class', () => {
    toast('hello world', 'success');
    const el = document.querySelector('.toast');
    expect(el).not.toBeNull();
    expect(el.classList.contains('toast-success')).toBe(true);
    expect(el.querySelector('.toast-msg').textContent).toBe('hello world');
  });

  it('escapes HTML in the message (XSS guard)', () => {
    toast('<script>alert(1)</script>', 'info');
    const el = document.querySelector('.toast .toast-msg');
    // Inner text should preserve the chars; innerHTML should be escaped
    expect(el.textContent).toContain('<script>');
    expect(el.innerHTML).not.toContain('<script>');
    expect(el.innerHTML).toContain('&lt;script&gt;');
  });

  it('returns a function that removes the toast on click', () => {
    toast('dismissable', 'info');
    expect(document.querySelectorAll('.toast').length).toBe(1);
    document.querySelector('.toast .toast-close').click();
    // Removal is animated (250ms); the .hide class flips immediately
    const el = document.querySelector('.toast');
    expect(el.classList.contains('hide')).toBe(true);
  });
});

describe('confirmDialog()', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('resolves true on confirm-button click', async () => {
    const promise = confirmDialog({ title: 'Sure?', confirmLabel: 'Yes' });
    const btn = document.querySelector('[data-role="confirm"]');
    expect(btn.textContent).toBe('Yes');
    btn.click();
    await expect(promise).resolves.toBe(true);
    // Modal should be removed from the DOM
    expect(document.querySelector('.modal-overlay')).toBeNull();
  });

  it('resolves false on cancel-button click', async () => {
    const promise = confirmDialog({ title: 'Sure?' });
    document.querySelector('[data-role="cancel"]').click();
    await expect(promise).resolves.toBe(false);
  });

  it('resolves false when Escape is pressed', async () => {
    const promise = confirmDialog({ title: 'Sure?' });
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    await expect(promise).resolves.toBe(false);
  });
});
