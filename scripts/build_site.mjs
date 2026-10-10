// Generate static, accessible MathJax formulas before Markdown can alter their TeX.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {Marked, Renderer} from 'marked';

const require = createRequire(import.meta.url);
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AssistiveMmlHandler} = require('mathjax-full/js/a11y/assistive-mml.js');
require('mathjax-full/js/input/tex/ams/AmsConfiguration.js');
require('mathjax-full/js/input/tex/boldsymbol/BoldsymbolConfiguration.js');
const adaptor = liteAdaptor();
AssistiveMmlHandler(RegisterHTMLHandler(adaptor));
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT = path.join(ROOT, '_site');
const key = path.basename(ROOT).split('-')[0];
assert(['ppo', 'grpo'].includes(key));
const topic = key.toUpperCase();
const repo = `https://github.com/zwdx/${key}-all-you-need-to-know`;
const companion = key === 'ppo' ? 'grpo' : 'ppo';
const read = name => fs.readFileSync(path.join(ROOT, name), 'utf8');
const escape = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const plain = text => text.replace(/\$`([^`]+)`\$/g, '$1').replace(/<[^>]+>/g, '').replace(/[*`#]/g, '').trim();
const chapters = fs.readdirSync(path.join(ROOT, 'docs')).filter(name => /^chapter-\d+\.md$/.test(name)).sort();
const pages = [{source:'README.md', file:'index.html', title:'项目首页'},
  {source:'docs/README.md', file:'guide.html', title:'阅读路线与符号表'},
  ...chapters.map(name => ({source:`docs/${name}`, file:name.replace('.md', '.html'),
    title:read(`docs/${name}`).match(/^# (.+)$/m)[1]}))];
const destinations = new Map(pages.map(p => [p.source, p.file]));
const search = [], files = new Map();
let totalMath = 0;

function linkTarget(href, source) {
  if (/^(?:[a-z]+:|\/\/)/i.test(href)) return href;
  const [location, fragment] = href.split('#');
  const target = location ? path.posix.normalize(path.posix.join(path.posix.dirname(source), decodeURIComponent(location))) : source;
  if (destinations.has(target)) return destinations.get(target) + (fragment ? '#' + fragment.replace(/^user-content-/, '') : '');
  return `${repo}/blob/main/${target}` + (fragment ? '#' + fragment : '');
}

function expectedMath(markdown) {
  const found = [], lines = markdown.split('\n');
  let fence, language, body;
  for (const line of lines) {
    const match = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
    if (match && !fence) { fence = match[1]; language = match[2].trim(); body = []; continue; }
    if (fence) {
      if (match && match[1][0] === fence[0] && match[1].length >= fence.length && !match[2].trim()) {
        if (language === 'math') found.push(body.join('\n').trim());
        fence = undefined;
      } else body.push(line);
      continue;
    }
    for (const token of line.matchAll(/\$`(?<math>[^`\n]+)`\$|(?<ticks>`+).*?\k<ticks>/g)) {
      if (token.groups.math !== undefined) found.push(token.groups.math);
    }
  }
  assert(!fence, 'Unclosed fence');
  return found;
}

for (const page of pages) {
  let markdown = read(page.source);
  // Generated Markdown already supplies previous/next links; the website supplies them in its frame.
  markdown = markdown.replace(/^\[(?:上一章|返回目录|项目首页)\][^\n]+\n\n(?:---\n\n)?/, '')
    .replace(/\n---\n\n\[(?:上一章|返回目录)\][^\n]+\n$/, '\n')
    .replace(/\*\*本章目录\*\*\n\n([\s\S]*?)\n\n/, '<details>\n<summary>本章目录</summary>\n\n$1\n\n</details>\n\n');
  const mathSource = [], errors = [];
  const svg = new SVG({fontCache:'global'});
  const input = new TeX({packages:['base', 'ams', 'boldsymbol'],
    formatError: (jax, error) => { errors.push(error.message); return jax.formatError(error); }});
  const document = mathjax.document('', {InputJax:input, OutputJax:svg, enableAssistiveMml:true});
  function equation(tex, display) {
    tex = tex.trim();
    mathSource.push(tex);
    const node = document.convert(tex, {display});
    assert(!errors.length, `${page.source}: ${tex}\n${errors.join('\n')}`);
    const output = adaptor.outerHTML(node);
    assert(!output.includes('data-mjx-error'));
    if (!display) return `<span class="math-inline" data-tex="${escape(tex)}">${output}</span>`;
    return `<div class="equation"><button class="copy-math" data-tex="${escape(tex)}" aria-label="复制这条公式的 LaTeX">复制公式</button><div class="equation-scroll" tabindex="0" aria-label="公式，可横向滚动">${output}</div></div>\n`;
  }
  let headingNumber = 0;
  const marked = new Marked({gfm:true, renderer:{
    code(token) {
      if (token.lang === 'math') return equation(token.text, true);
      return Renderer.prototype.code.call(this, token);
    },
    link(token) { return Renderer.prototype.link.call(this, {...token, href:linkTarget(token.href, page.source)}); },
    table(token) { return '<div class="table-scroll">' + Renderer.prototype.table.call(this, token) + '</div>\n'; },
    heading(token) { return `<h${token.depth} id="heading-${++headingNumber}">${this.parser.parseInline(token.tokens)}</h${token.depth}>\n`; }
  }, extensions:[{
    name:'githubMath', level:'inline', start:src => src.indexOf('$`'),
    tokenizer(src) {
      const match = /^\$`([^`\n]+)`\$/.exec(src);
      if (match) return {type:'githubMath', raw:match[0], text:match[1]};
    },
    renderer:token => equation(token.text, false)
  }]});
  const body = marked.parse(markdown);
  // Check every occurrence, including answers and table cells, before adding navigation formulas.
  assert.deepEqual([...mathSource].sort(), expectedMath(markdown).map(tex => tex.trim()).sort(), `${page.source}: Formula lost or changed during Markdown parsing`);
  totalMath += mathSource.length;
  const entries = [...markdown.matchAll(/^<a id="([^"]+)"><\/a>\n\n(#{1,3}) (.+)$/gm)];
  const sectionLinks = entries.filter(m => m[2].length === 2).map(m =>
    `<li><a href="#${m[1]}">${marked.parseInline(m[3])}</a></li>`).join('');
  const menu = pages.map(item => {
    const current = item.file === page.file;
    return `<li><a href="${item.file}"${current ? ' aria-current="page"' : ''}>${escape(plain(item.title))}</a>${current && sectionLinks ? `<ul class="section-links">${sectionLinks}</ul>` : ''}</li>`;
  }).join('');
  if (page.source !== 'README.md') {
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const text = markdown.slice(entry.index + entry[0].length, entries[i+1]?.index ?? markdown.length);
      search.push({title:plain(entry[3]), url:`${page.file}#${entry[1]}`,
        text:plain(text).replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').replace(/\s+/g, ' ')});
    }
  }
  const position = pages.indexOf(page);
  const previous = position > 1 ? pages[position-1] : null;
  const next = pages[position+1];
  const chapterNav = `<nav class="chapter-nav" aria-label="章节翻页">${previous ? `<a href="${previous.file}">← ${escape(plain(previous.title))}</a>` : ''}${next ? `<a href="${next.file}">${escape(plain(next.title))} →</a>` : ''}</nav>`;
  const styles = adaptor.outerHTML(svg.styleSheet(document));
  const cache = `<svg class="font-cache" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">${adaptor.outerHTML(svg.fontCache.getCache())}</svg>`;
  files.set(page.file, `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escape(plain(page.title))} · ${topic} 学习教程</title><meta name="description" content="${topic} 中文系统教程：数学推导、算法理解、工程实践与章末练习。">
<link rel="stylesheet" href="site.css">${styles}<script src="search-index.js" defer></script><script src="site.js" defer></script></head>
<body>${cache}<a class="skip" href="#main">跳到正文</a>
<header class="topbar"><div class="topbar-inner"><a class="brand" href="index.html">${topic}<small>ALL YOU NEED TO KNOW</small></a>
<div class="top-links"><button id="open-search" type="button" aria-haspopup="dialog">搜索</button><a class="companion" href="https://zwdx.github.io/${companion}-all-you-need-to-know/">${companion.toUpperCase()} 教程</a><a href="${repo}">GitHub</a></div></div></header>
<div class="layout"><aside class="sidebar"><details id="book-menu" open><summary>章节目录</summary><nav aria-label="章节目录"><ul>${menu}</ul></nav></details></aside>
<main id="main" tabindex="-1"><p class="eyebrow">${topic} / 中文系统教程</p><article>${body}</article>${chapterNav}
<footer>作者 <a href="https://github.com/zwdx">zwdx</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · <a href="${repo}/issues">报告勘误</a> · <a href="${repo}/blob/main/${encodeURI(page.source)}">查看 Markdown 源文档</a></footer></main></div>
<dialog id="search-dialog" aria-labelledby="search-title"><div class="search-head"><h2 id="search-title">搜索教程</h2><button id="close-search" type="button" aria-label="关闭搜索">关闭</button></div><label for="search-input">章节名、概念或公式关键词</label><input id="search-input" type="search" autocomplete="off" placeholder="例如：GAE、长度偏差、重要性采样"><p id="search-status" role="status">输入章节名、概念或公式关键词。</p><ul id="search-results"></ul></dialog>
</body></html>\n`);
}
files.set('site.css', read('scripts/site.css'));
files.set('site.js', read('scripts/site.js'));
files.set('search-index.js', 'const tutorialSearchIndex = ' + JSON.stringify(search) + ';\n');
files.set('.nojekyll', '');

// Fail builds on broken local links or missing SVG glyphs, without a browser/framework dependency.
for (const [name, html] of files) {
  if (!name.endsWith('.html')) continue;
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
  assert.equal(ids.length, new Set(ids).size, `${name}: Duplicate ID`);
  for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const href = match[1];
    if (/^(?:[a-z]+:|\/\/)/i.test(href)) continue;
    const [file, fragment] = href.split('#');
    const target = file || name;
    assert(files.has(target), `${name}: Missing ${target}`);
    if (fragment) assert(files.get(target).includes(`id="${fragment}"`), `${name}: Missing #${fragment} in ${target}`);
  }
  assert(!html.includes('data-mjx-error'));
}
for (const entry of search) {
  const [file, fragment] = entry.url.split('#');
  assert(files.get(file).includes(`id="${fragment}"`));
}
if (process.argv.includes('--check')) {
  for (const [name, content] of files) assert.equal(fs.readFileSync(path.join(OUTPUT, name), 'utf8'), content, `${name}: Run npm run build`);
} else {
  fs.mkdirSync(OUTPUT, {recursive:true});
  for (const [name, content] of files) fs.writeFileSync(path.join(OUTPUT, name), content);
}
console.log(`${topic}: ${pages.length} pages, ${totalMath} preserved formulas, ${search.length} searchable sections; links and SVG references valid`);
