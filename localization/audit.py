#!/usr/bin/env python3
"""对本地化改动做可复现的结构检查；不把结构通过等同于语义翻译正确。

在仓库根目录运行：python localization/audit.py --output /tmp/localization-audit.json
默认对照本地化开始时的提交；不会执行项目代码、修改文件或联网。
"""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata
import warnings

BASELINE = '0edf4bf45fbfb052d5f5760929f28998e1631de3'
ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings('ignore', category=SyntaxWarning)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.DEVNULL)


class TextMask(ast.NodeTransformer):
    """保留计算结构与数值，忽略文档字符串及字符串内容，另行列出字符串变更。"""
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value='<text>'), node)
        return node


def signature(source):
    tree = ast.parse(source)
    strings = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    # 翻译 f-string 字面段时可能合并相邻字面量，因此只比较表达式和其格式规格。
    return ast.dump(TextMask().visit(tree), include_attributes=False), strings


def slug(text):
    text = re.sub(r'<[^>]*>', '', text).strip().lower()
    text = text.replace('`', '')
    text = ''.join(c for c in text if c in ' -_' or unicodedata.category(c)[0] in 'LN')
    return text.replace(' ', '-')


def anchors(text):
    counts = collections.Counter()
    out = set(re.findall(r'\bid=["\']([^"\']+)', text))
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith(('```', '~~~')):
            fenced = not fenced
        if not fenced and re.match(r'^#{1,6}\s', line):
            key = slug(re.sub(r'^#+\s+', '', line))
            n = counts[key]
            counts[key] += 1
            out.add(key + (f'-{n}' if n else ''))
    return out


def link_errors(path, text, read):
    errs = set()
    # 不将 fenced 示例当作实际 Markdown 导航。
    text = re.sub(r'```.*?```|~~~.*?~~~', '', text, flags=re.S)
    for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)', text):
        if re.match(r'^[A-Za-z][\w+.-]*:', target):
            continue
        filename, sep, fragment = target.partition('#')
        resolved = (path.parent / filename).resolve() if filename else path
        try:
            relative = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            errs.add(target)
            continue
        body = read(relative)
        if body is None:
            errs.add(target)
        elif sep and fragment and resolved.suffix == '.md' and fragment not in anchors(body):
            errs.add(target)
    return errs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default=BASELINE)
    parser.add_argument('--output', default='/tmp/30u30-localization-audit.json')
    args = parser.parse_args()
    files = git('ls-tree', '-r', '--name-only', args.baseline).decode().splitlines()
    previous = {}
    directories = {str(Path(f).parent) for f in files}
    directories |= {str(parent) for f in files for parent in Path(f).parents}
    def old(path):
        if path in directories:
            return ''
        if path not in previous:
            try:
                previous[path] = git('show', f'{args.baseline}:{path}').decode()
            except subprocess.CalledProcessError:
                previous[path] = None
        return previous[path]
    def current(path):
        p = ROOT / path
        return p.read_text() if p.is_file() else ('' if p.is_dir() else None)
    result = {'baseline': args.baseline, 'files': [], 'failures': [],
              'baseline_syntax_errors': [], 'new_link_errors': [],
              'preexisting_link_errors': [], 'string_changes': []}
    for name in files:
        p = ROOT / name
        if p.suffix not in {'.py', '.md', '.ipynb', '.html', '.js', '.css', '.txt'}:
            continue
        before = old(name)
        after = current(name)
        if after is None:
            result['failures'].append({'path': name, 'reason': '原文件缺失'})
            continue
        changed = before != after
        row = {'path': name, 'changed': changed,
               'has_chinese': bool(re.search(r'[\u3400-\u9fff]', after)),
               'sha256': hashlib.sha256(after.encode()).hexdigest()}
        result['files'].append(row)
        if p.suffix == '.py':
            try:
                a, sa = signature(before)
            except SyntaxError as e:
                result['baseline_syntax_errors'].append({'path': name, 'line': e.lineno, 'message': e.msg})
                continue
            try:
                b, sb = signature(after)
                if a != b:
                    result['failures'].append({'path': name, 'reason': '非文本 AST 发生变化，需要审查'})
                if sa != sb:
                    result['string_changes'].append({'path': name, 'before': sa, 'after': sb})
            except SyntaxError as e:
                result['failures'].append({'path': name, 'reason': f'新增语法错误：{e}'})
        elif p.suffix == '.ipynb':
            a, b = json.loads(before.lstrip('\ufeff')), json.loads(after.lstrip('\ufeff'))
            if len(a['cells']) != len(b['cells']):
                result['failures'].append({'path': name, 'reason': 'Notebook 单元数量变化'})
                continue
            for i, (x, y) in enumerate(zip(a['cells'], b['cells'])):
                if x['cell_type'] != y['cell_type']:
                    result['failures'].append({'path': name, 'cell': i, 'reason': '单元类型变化'})
                if x['cell_type'] == 'code':
                    if x.get('outputs') != y.get('outputs'):
                        result['failures'].append({'path': name, 'cell': i, 'reason': '历史输出变化，需要审查'})
                    try:
                        if signature(''.join(x['source']))[0] != signature(''.join(y['source']))[0]:
                            result['failures'].append({'path': name, 'cell': i, 'reason': 'Notebook 非文本 AST 变化'})
                    except SyntaxError:
                        # 含 Notebook magic 的单元不能用 Python AST 证明；单独列为未检查。
                        row.setdefault('unparsed_cells', []).append(i)
        elif p.suffix == '.md':
            previous_errors = link_errors(p, before, old)
            now_errors = link_errors(p, after, current)
            for target in sorted(now_errors - previous_errors):
                result['new_link_errors'].append({'path': name, 'target': target})
            for target in sorted(now_errors & previous_errors):
                result['preexisting_link_errors'].append({'path': name, 'target': target})
            old_math = collections.Counter(re.findall(r'\$\$.*?\$\$|(?<!\$)\$(?!\$)[^\n$]+\$', before, re.S))
            new_math = collections.Counter(re.findall(r'\$\$.*?\$\$|(?<!\$)\$(?!\$)[^\n$]+\$', after, re.S))
            missing = list((old_math - new_math).elements())
            if missing:
                row['math_fragments_to_review'] = missing
        if (p.name == 'requirements.txt' or p.name.endswith('lock')) and changed:
            result['failures'].append({'path': name, 'reason': '依赖声明变更'})
    result['summary'] = {'tracked_text_files': len(result['files']),
                         'modified_files': sum(f['changed'] for f in result['files']),
                         'structural_failures': len(result['failures']),
                         'baseline_syntax_errors': len(result['baseline_syntax_errors']),
                         'new_local_link_errors': len(result['new_link_errors'])}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['summary'], ensure_ascii=False))
    print('字符串变化仅列为待审清单；汉字存在率与文件改动率不代表翻译完成率。')
    raise SystemExit(bool(result['failures'] or result['new_link_errors']))


if __name__ == '__main__':
    main()
