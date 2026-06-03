# -*- coding: utf-8 -*-
"""
document_parsing.py
===================
文档解析模块。

职责：
1. 调用 LlamaParse 将学术 PDF 高精度解析为 Markdown（保留标题、表格、公式结构）。
2. 使用正则提取 Markdown 标题层级（H1~H6），构建文档大纲树。
3. 将 Markdown、标题树、章节切分结果持久化到 output/ 目录（JSON / MD）。

输出的核心数据结构 `sections`（列表），每个元素：
{
    "section_id": "doc123::s0003",   # 全局唯一
    "doc_id":     "doc123",
    "level":      2,                  # 标题级别 1~6
    "title":      "3.2 Method",
    "path":       ["3 Approach", "3.2 Method"],  # 从根到当前的标题路径
    "content":    "<该标题下、下一个同级或更高级标题之前的正文>",
    "order":      3                   # 章节在文中的出现顺序
}
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import config


# Markdown ATX 标题正则：捕获 # 数量（级别）与标题文本
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$", re.MULTILINE)


class DocumentParser:
    """学术 PDF -> Markdown -> 标题层级章节 的解析器。"""

    def __init__(self,
                 llama_cloud_api_key: Optional[str] = None,
                 output_dir: Optional[Path] = None):
        self.api_key = llama_cloud_api_key or config.LLAMA_CLOUD_API_KEY
        self.output_dir = Path(output_dir or config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 第一步：PDF -> Markdown（LlamaParse）
    # ------------------------------------------------------------------ #
    def parse_pdf_to_markdown(self, pdf_path: str) -> str:
        """调用 LlamaParse 把 PDF 转为 Markdown。

        若环境无网络或未配置 key，可将已解析好的 .md 放入 output/ 走缓存逻辑。
        """
        pdf_path = str(pdf_path)
        doc_id = self._make_doc_id(pdf_path)
        md_cache = self.output_dir / f"{doc_id}.md"

        # 命中缓存直接返回，避免重复消耗 LlamaParse 额度
        if md_cache.exists():
            return md_cache.read_text(encoding="utf-8")

        # 延迟导入，避免无该依赖时影响其它模块
        from llama_parse import LlamaParse

        parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",      # 关键：输出 Markdown 以保留标题层级
            parse_mode="parse_page_with_agent",
            # 针对学术论文的提示，提升公式 / 表格 / 多栏排版的还原度
            user_prompt=(
                "This is an academic paper. Preserve the section heading "
                "hierarchy using Markdown headers (#, ##, ###). Keep tables "
                "as Markdown tables and keep math formulas in LaTeX."
            ),
            verbose=True,
        )

        documents = parser.load_data(pdf_path)
        markdown = "\n\n".join(d.text for d in documents)

        md_cache.write_text(markdown, encoding="utf-8")
        return markdown

    # ------------------------------------------------------------------ #
    # 第二步：正则提取标题层级，构建章节
    # ------------------------------------------------------------------ #
    def extract_sections(self, markdown: str, doc_id: str) -> List[Dict]:
        """根据 Markdown 标题把全文切成"以标题为界"的章节，并保留层级路径。"""
        matches = list(_HEADING_RE.finditer(markdown))

        sections: List[Dict] = []
        if not matches:
            # 没有任何标题：整篇作为一个根章节
            sections.append({
                "section_id": f"{doc_id}::s0000",
                "doc_id": doc_id,
                "level": 1,
                "title": "Full Document",
                "path": ["Full Document"],
                "content": markdown.strip(),
                "order": 0,
            })
            return sections

        # 维护一个"当前标题栈"，用于回溯 path
        stack: List[Dict] = []  # 每个元素 {"level", "title"}

        for idx, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2).strip()

            # 正文范围：本标题末尾 -> 下一标题开头
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
            content = markdown[start:end].strip()

            # 维护层级栈：弹出所有 >= 当前级别的标题
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            stack.append({"level": level, "title": title})
            path = [s["title"] for s in stack]

            sections.append({
                "section_id": f"{doc_id}::s{idx:04d}",
                "doc_id": doc_id,
                "level": level,
                "title": title,
                "path": path,
                "content": content,
                "order": idx,
            })

        return sections

    def build_heading_tree(self, sections: List[Dict]) -> List[Dict]:
        """把扁平 sections 组装成嵌套树，用于前端大纲展示与调试。"""
        root: List[Dict] = []
        stack: List[Dict] = []  # (level, node)

        for sec in sections:
            node = {
                "section_id": sec["section_id"],
                "level": sec["level"],
                "title": sec["title"],
                "children": [],
            }
            while stack and stack[-1]["level"] >= sec["level"]:
                stack.pop()
            if stack:
                stack[-1]["node"]["children"].append(node)
            else:
                root.append(node)
            stack.append({"level": sec["level"], "node": node})
        return root

    # ------------------------------------------------------------------ #
    # 顶层入口
    # ------------------------------------------------------------------ #
    def parse(self, pdf_path: str) -> Dict:
        """完整解析流程，返回结构化结果并落盘。"""
        doc_id = self._make_doc_id(pdf_path)
        markdown = self.parse_pdf_to_markdown(pdf_path)
        sections = self.extract_sections(markdown, doc_id)
        tree = self.build_heading_tree(sections)

        result = {
            "doc_id": doc_id,
            "source": os.path.basename(pdf_path),
            "markdown_path": str(self.output_dir / f"{doc_id}.md"),
            "num_sections": len(sections),
            "sections": sections,
            "heading_tree": tree,
        }

        # 持久化结构化结果
        json_path = self.output_dir / f"{doc_id}.sections.json"
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_doc_id(pdf_path: str) -> str:
        """基于文件名 + 内容指纹生成稳定 doc_id（便于缓存复用）。"""
        name = Path(pdf_path).stem
        safe = re.sub(r"[^0-9A-Za-z_-]", "_", name)[:32]
        try:
            with open(pdf_path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()[:8]
        except FileNotFoundError:
            digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        return f"{safe}_{digest}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python document_parsing.py <path/to/paper.pdf>")
        raise SystemExit(1)

    parser = DocumentParser()
    res = parser.parse(sys.argv[1])
    print(f"解析完成 doc_id={res['doc_id']}, 共 {res['num_sections']} 个章节")
    for s in res["sections"][:10]:
        print("  " * (s["level"] - 1) + f"- [{s['level']}] {s['title']}")
