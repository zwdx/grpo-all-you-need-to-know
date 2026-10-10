// Run the actual browser script with a small DOM stub; no browser or test dependency required.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

function run(saved, unavailable = false) {
  const elements = new Map(), styles = new Map(), storage = new Map([['ppo-grpo-reading', saved]]);
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, {value:'', textContent:selector === '#search-data' ? '[]' : '',
      listeners:{}, addEventListener(type, callback) { this.listeners[type] = callback; },
      replaceChildren() {}, showModal() { this.open = true; }, close() { this.open = false; }, focus() {}});
    return elements.get(selector);
  };
  vm.runInNewContext(fs.readFileSync(new URL('./site.js', import.meta.url), 'utf8'), {
    document:{querySelector:element, querySelectorAll:() => [], addEventListener() {},
      documentElement:{style:{setProperty:(key, value) => styles.set(key, String(value))}}},
    window:{addEventListener() {}}, matchMedia:() => ({matches:true, addEventListener() {}}),
    localStorage:{getItem:key => { if (unavailable) throw Error('Storage disabled'); return storage.get(key); },
      setItem:(key, value) => { if (unavailable) throw Error('Storage disabled'); storage.set(key, value); }}
  });
  return {element, styles, storage};
}

for (const saved of [undefined, 'null', '{broken', '{"size":1000,"spacing":0}', '{"size":"24","spacing":"2.1"}']) {
  const {styles} = run(saved);
  assert.equal(styles.get('--reading-size'), '17px');
  assert.equal(styles.get('--reading-spacing'), '1.85');
}
const {element, styles, storage} = run('{"size":24,"spacing":2.1}');
assert.equal(styles.get('--reading-size'), '24px');
assert.equal(element('#reading-size-value').textContent, '24 px');
assert.equal(styles.get('--reading-spacing'), '2.1');
element('#reading-size').value = '14';
element('#reading-spacing').value = '1.6';
element('#reading-size').listeners.input();
assert.deepEqual(JSON.parse(storage.get('ppo-grpo-reading')), {size:14, spacing:1.6});
assert.equal(styles.get('--reading-size'), '14px');
element('#reset-reading').listeners.click();
assert.deepEqual(JSON.parse(storage.get('ppo-grpo-reading')), {size:17, spacing:1.85});
element('#open-reading').listeners.click();
assert.equal(element('#reading-dialog').open, true);
element('#close-reading').listeners.click();
assert.equal(element('#reading-dialog').open, false);
const disabled = run(null, true);
disabled.element('#reset-reading').listeners.click();
disabled.element('#open-search').listeners.click();
assert.equal(disabled.element('#search-dialog').open, true);
console.log('Reading settings: defaults, bounds, persistence, reset and disabled storage passed');
