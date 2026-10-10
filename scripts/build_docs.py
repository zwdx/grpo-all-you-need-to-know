"""从完整文档生成网站章节输入与在线目录；--check 检查同步与链接。"""
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = re.compile(r'^<a (?:name|id)="[^"]+"></a>\n\n', re.M)
TOC = re.compile(r'\n<!-- toc:start -->\n.*?<!-- toc:end -->\n\n', re.S)
CHAPTER = re.compile(r'^第\s*(\d+)\s*章')
HASH = '#user-content-'  # ponytail: GitHub prefixes HTML IDs; use '#' for a renderer that preserves IDs.
INLINE = re.compile(r'\$`(?P<math>[^`\n]+)`\$|(?P<ticks>`+).*?(?P=ticks)')


def headings(text):
    result, fence, chapter, used = [], None, 0, set()
    for index, line in enumerate(text.splitlines()):
        match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if match:
            token, suffix = match.groups()
            if fence is None:
                if suffix.strip() == 'math':
                    assert not line.startswith(' '), f'第 {index + 1} 行：math 围栏须位于顶层，不能缩进进列表'
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence) and not suffix.strip():
                fence = None
            continue
        if fence is None:
            for item in INLINE.finditer(line):
                tex = item['math']
                if tex is not None:
                    assert not re.search(r'[<>]', tex), f'第 {index + 1} 行：行内公式须用 \\lt / \\gt 避免 HTML 实体转义'
                    assert not (line.startswith('|') and '|' in tex), f'第 {index + 1} 行：表格公式须用 \\mid、\\vert 或 \\Vert 避免切列与反斜杠丢失'
        match = re.match(r'^(#{1,6})\s+(.+?)\s*$', line) if fence is None else None
        if not match:
            continue
        level, title = len(match[1]), match[2]
        ch = CHAPTER.match(title) if level == 1 else None
        if ch:
            chapter = int(ch[1])
        number = re.match(r'^(\d+(?:\.\d+)+)\s', title)
        anchor = f'chapter-{chapter}' if ch else (
            's' + number[1].replace('.', '-') if number else f'h{len(result) + 1}')
        assert anchor not in used, f'重复章节编号：{title}'
        used.add(anchor)
        result.append((index, level, title, anchor, chapter))
    assert fence is None, '代码围栏未闭合'
    return result


def annotate(lines, entries, offset=0):
    anchors = {entry[0] - offset: entry[3] for entry in entries}
    return '\n'.join(('<a id="' + anchors[i] + '"></a>\n\n' if i in anchors else '') + line
                     for i, line in enumerate(lines)).rstrip() + '\n'


def toc(entries, target=''):
    chapters = [e for e in entries if e[1] == 1 and CHAPTER.match(e[2])]
    links = [f'- [{e[2]}]({target}{HASH}{e[3]})' for e in chapters]
    sections = [f'{"  " if e[1] == 3 else ""}- [{e[2]}]({target}{HASH}{e[3]})'
                for e in entries if e[4] and e[1] in (2, 3)]
    return '\n'.join(links + ['', '<details>', '<summary>展开小节目录</summary>', '']
                     + sections + ['', '</details>', ''])


def fold_answers(text):
    lines, out, answer, fence = text.splitlines(), [], False, False
    for line in lines + ['']:
        if line.startswith('```'):
            fence = not fence
        if not fence and re.match(r'^(?:\*\*)?(?:参考答案|答案)(?:\*\*)?\s*[：:]', line):
            assert not answer
            out += ['<details>', '<summary>查看参考答案</summary>', '']
            answer = True
        if answer and not line.strip():
            out += ['', '</details>']
            answer = False
        out.append(line)
    return '\n'.join(out).rstrip() + '\n'


def generate(source):
    clean = TOC.sub('', ANCHOR.sub('', source.read_text(encoding='utf-8')))
    lines, entries = clean.splitlines(), headings(clean)
    chapters = [e for e in entries if e[1] == 1 and CHAPTER.match(e[2])]
    assert chapters and [e[4] for e in chapters] == list(range(1, len(chapters) + 1))
    body = annotate(lines, entries)
    first = entries[0]
    intro_links = [f'[{e[2]}]({HASH}{e[3]})' for e in entries if e[2] in ('阅读路线', '全文通用符号约定')]
    full_toc = '<!-- toc:start -->\n[返回项目首页](README.md) · ' + ' · '.join(intro_links)
    full_toc += '\n\n## 可点击目录\n\n' + toc(entries) + '\n<!-- toc:end -->\n\n'
    title_line = '# ' + first[2] + '\n'
    body = body.replace(title_line, title_line + '\n' + full_toc, 1)
    files = {source.name: body}
    intro_entries = [e for e in entries if e[0] < chapters[0][0]]
    guide = '[项目首页](../README.md)\n\n'
    guide += annotate(lines[:chapters[0][0]], intro_entries)
    guide += '\n## 章节目录\n\n' + '\n'.join(f'- [{e[2]}](chapter-{e[4]:02d}.md)' for e in chapters) + '\n'
    files['docs/README.md'] = guide
    rows = ['| 章节 | 阅读入口 |', '|---|---|']
    for i, entry in enumerate(chapters):
        start, _, title, anchor, number = entry
        end = chapters[i + 1][0] if i + 1 < len(chapters) else len(lines)
        chapter_entries = [e for e in entries if start <= e[0] < end]
        nav = ['[返回目录](../README.md)', '[阅读路线与符号表](README.md)']
        if i:
            nav.insert(0, f'[上一章](chapter-{number - 1:02d}.md)')
        if i + 1 < len(chapters):
            nav.append(f'[下一章](chapter-{number + 1:02d}.md)')
        navigation = ' · '.join(nav)
        local_toc = '\n'.join(f'- [{e[2]}]({HASH}{e[3]})' for e in chapter_entries if e[1] == 2)
        chapter_body = fold_answers(annotate(lines[start:end], chapter_entries, start))
        chapter_body = chapter_body.replace('# ' + title + '\n', '# ' + title + '\n\n**本章目录**\n\n' + local_toc + '\n', 1)
        files[f'docs/chapter-{number:02d}.md'] = navigation + '\n\n---\n\n' + chapter_body + '\n---\n\n' + navigation + '\n'
        rows.append(f'| {title} | [在线阅读](https://zwdx.github.io/{ROOT.name}/chapter-{number:02d}.html) |')
    readme = (ROOT / 'README.md').read_text(encoding='utf-8').replace('.md#s', '.md' + HASH + 's')
    files['README.md'] = re.sub(r'<!-- chapters:start -->.*?<!-- chapters:end -->',
                               lambda _: '<!-- chapters:start -->\n' + '\n'.join(rows) + '\n<!-- chapters:end -->', readme, flags=re.S)
    return files


def check_links(files):
    count = 0
    for name, text in files.items():
        ids = re.findall(r'<a id="([^"]+)"></a>', text)
        assert len(ids) == len(set(ids)), f'{name}: 锚点重复'
        assert text.count('<details>') == text.count('</details>') == text.count('<summary>'), f'{name}: 折叠标签未配对'
        headings(text)
        visible = re.sub(r'(?ms)^```.*?^```[^\n]*|\$\$.*?\$\$|\$[^$\n]*\$', '', text)
        for link in re.findall(r'\]\(([^\s()]+)\)', visible):
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = ((ROOT / name).parent / unquote(url.path)).resolve() if url.path else ROOT / name
            assert target.is_relative_to(ROOT), f'{name}: 链接越过仓库边界：{link}'
            assert target.is_file(), f'{name}: 文件不存在：{link}'
            if url.fragment:
                assert url.fragment.startswith('user-content-'), f'{name}: 缺少 GitHub 锚点前缀：{link}'
                anchor = unquote(url.fragment).removeprefix('user-content-')
                assert '<a id="' + anchor + '"></a>' in target.read_text(encoding='utf-8'), f'{name}: 锚点不存在：{link}'
            count += 1
    return count


if __name__ == '__main__':
    sources = list(ROOT.glob('*系统学习文档.md'))
    assert len(sources) == 1, '仓库根目录须有且只有一份系统学习文档'
    files = generate(sources[0])
    if '--check' in sys.argv:
        for name, text in files.items():
            assert (ROOT / name).read_text(encoding='utf-8') == text, f'{name}: 请先执行 python scripts/build_docs.py'
    else:
        for name, text in files.items():
            path = ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8', newline='\n')
    count = check_links({str(p.relative_to(ROOT)).replace('\\', '/'): p.read_text(encoding='utf-8')
                         for p in [*ROOT.glob('*.md'), *(ROOT / 'docs').rglob('*.md')]})
    print(f'OK: {len(files)} generated documents, {count} valid local links')
