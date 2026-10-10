const book = document.querySelector('#book-menu');
const wide = matchMedia('(min-width: 901px)');
const fitMenu = () => { book.open = wide.matches; };
fitMenu();
wide.addEventListener('change', fitMenu);

document.querySelectorAll('.copy-math').forEach(button => {
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(button.dataset.tex);
      button.textContent = '已复制';
      setTimeout(() => { button.textContent = '复制公式'; }, 1800);
    } catch {
      button.textContent = '复制未成功';
    }
  });
});

const fitInlineMath = () => {
  document.querySelectorAll('.math-inline').forEach(element => {
    const scroll = element.querySelector('mjx-container').getBoundingClientRect().width > element.clientWidth + 1;
    element.classList.toggle('is-wide', scroll);
    if (scroll) element.setAttribute('tabindex', '0');
    else element.removeAttribute('tabindex');
  });
};
fitInlineMath();
window.addEventListener('resize', fitInlineMath);

const reading = document.querySelector('#reading-dialog');
const size = document.querySelector('#reading-size');
const spacing = document.querySelector('#reading-spacing');
const readingKey = 'ppo-grpo-reading';
const applyReading = value => {
  const fontSize = Number.isInteger(value?.size) && value.size >= 14 && value.size <= 24 ? value.size : 17;
  const lineHeight = [1.6, 1.85, 2.1].includes(value?.spacing) ? value.spacing : 1.85;
  size.value = fontSize;
  spacing.value = lineHeight;
  document.querySelector('#reading-size-value').textContent = `${fontSize} px`;
  document.documentElement.style.setProperty('--reading-size', `${fontSize}px`);
  document.documentElement.style.setProperty('--reading-spacing', lineHeight);
  fitInlineMath();
  return {size:fontSize, spacing:lineHeight};
};
let savedReading;
try { savedReading = JSON.parse(localStorage.getItem(readingKey)); } catch {}
applyReading(savedReading);
const saveReading = value => {
  const settings = applyReading(value);
  try { localStorage.setItem(readingKey, JSON.stringify(settings)); } catch {}
};
for (const control of [size, spacing]) control.addEventListener('input', () => saveReading({size:Number(size.value), spacing:Number(spacing.value)}));
document.querySelector('#open-reading').addEventListener('click', () => reading.showModal());
document.querySelector('#close-reading').addEventListener('click', () => reading.close());
document.querySelector('#reset-reading').addEventListener('click', () => saveReading());

const dialog = document.querySelector('#search-dialog');
const input = document.querySelector('#search-input');
const results = document.querySelector('#search-results');
const status = document.querySelector('#search-status');
const tutorialSearchIndex = JSON.parse(document.querySelector('#search-data').textContent);
results.addEventListener('click', event => { if (event.target.closest('a')) dialog.close(); });
const openSearch = () => { dialog.showModal(); input.focus(); };
document.querySelector('#open-search').addEventListener('click', openSearch);
document.querySelector('#close-search').addEventListener('click', () => dialog.close());
document.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    if (!dialog.open) openSearch();
  }
});
input.addEventListener('input', () => {
  const query = input.value.trim().toLowerCase();
  results.replaceChildren();
  if (!query) { status.textContent = '输入章节名、概念或公式关键词。'; return; }
  const words = query.split(/\s+/);
  const matches = tutorialSearchIndex.filter(item => words.every(word => (item.title + ' ' + item.text).toLowerCase().includes(word)));
  status.textContent = matches.length ? `找到 ${matches.length} 个小节，显示前 20 项。` : '未找到匹配的小节。';
  for (const item of matches.slice(0, 20)) {
    const li = document.createElement('li');
    const link = document.createElement('a');
    link.href = item.url;
    link.textContent = item.title;
    const snippet = document.createElement('p');
    const position = item.text.toLowerCase().indexOf(words[0]);
    snippet.textContent = item.text.slice(Math.max(0, position - 35), Math.max(0, position - 35) + 130);
    li.append(link, snippet);
    results.append(li);
  }
});
