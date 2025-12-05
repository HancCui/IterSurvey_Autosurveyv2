import subprocess
import tempfile
import os


# Dependency: playwright
import asyncio
from playwright.async_api import async_playwright

async def render_mermaid_with_playwright(mermaid_code: str, output_file: str):
    """
    使用 Playwright 和无头浏览器将 Mermaid 代码渲染成图片。
    """
    # 构造一个简单的 HTML 页面来承载 Mermaid 图表
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Mermaid Render</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.2.4/mermaid.min.js"></script>
    </head>
    <body>
        <div class="mermaid">
            {mermaid_code}
        </div>
        <script>mermaid.initialize({{ startOnLoad: true, theme:'neutral' }});</script>
    </body>
    </html>
    """

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # 直接设置页面内容
        await page.set_content(html_content)

        # 定位到渲染出的 SVG 元素
        diagram_element = await page.query_selector(".mermaid > svg")

        if diagram_element:
            # 对该元素截图并保存
            await diagram_element.screenshot(path=output_file)
            print(f"图片已成功保存到: {output_file}")
        else:
            print("错误: 未能渲染或找到 Mermaid 图表元素。")

        await browser.close()

# --- 使用示例 ---
my_diagram = """
graph LR
    subgraph "起点"
        A["<b>早期统计模型 (n-gram)</b><br>[1409.0473v7]"]
    end

    subgraph "早期神经网络时代"
        B["<b>RNN & LSTM 网络</b><br>[1609.08144v2]"]
    end

    subgraph "革命性转折"
        C["<b>Transformer 架构</b><br>[1706.03762]"]
    end

    subgraph "Transformer 核心组件"
        D["<b>自注意力机制</b><br>[1810.04805]"]
        E["<b>多头注意力</b><br>[1905.09418]"]
        F["<b>位置编码 (RoPE)</b><br>[2104.09864]"]
    end

    subgraph "大型语言模型 (LLM) 时代"
        G["<b>规模化定律</b><br>[2001.08361]"]
        H["<b>涌现能力</b><br>[2203.02155]"]
        I["<b>训练范式 (预训练+微调)</b><br>[2005.14165]"]
        J["<b>代表性模型: BERT</b><br>[1810.04805]"]
        K["<b>代表性模型: GPT-3</b><br>[2005.14165]"]
    end

    A --> B
    B --> C

    C ==> D
    C ==> E
    C ==> F

    C --> G
    G --> H
    C --> I
    I --> J
    I --> K
"""

# 运行异步函数
asyncio.run(render_mermaid_with_playwright(my_diagram, 'my_diagram_playwright.png'))