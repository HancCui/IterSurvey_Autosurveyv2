import asyncio
from datetime import date, timedelta
from arxiv_crawler import ArxivScraper  # Assuming ArxivScraper is in arxiv_crawler.py or similar

# --- 配置参数 ---
# 计算日期范围：从今天起往前推7天
today = date.today()
start_date = today - timedelta(days=7)

# 要搜索的关键词 (OR 关系)
keywords = [
    "LLM", "LLMs", "language model", "language models",
    "multimodal", "finetuning", "transformer", "transformers",
    "agent", "RAG", "retrieval augmented generation"
]

# 要保留的分类 (白名单)
categories_whitelist = ["cs.AI", "cs.LG", "cs.CL"]

# 要排除的分类 (黑名单) - 这里留空，可以根据需要添加
categories_blacklist = []

# 翻译目标语言 ('zh-CN' 表示中文，设为 None 或 False 则不翻译)
translate_to = 'zh-CN'

# 代理 (如果需要的话，例如 'http://127.0.0.1:7890')
proxy = None
# --- 配置结束 ---

print(f"开始爬取论文...")
print(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {today.strftime('%Y-%m-%d')}")
print(f"关键词: {', '.join(keywords)}")
print(f"领域白名单: {', '.join(categories_whitelist)}")

# 创建 ArxivScraper 实例
scraper = ArxivScraper(
    date_from=start_date.strftime("%Y-%m-%d"),
    date_until=today.strftime("%Y-%m-%d"),
    optional_keywords=keywords,
    category_whitelist=categories_whitelist,
    category_blacklist=categories_blacklist,
    trans_to=translate_to,
    proxy=proxy
)

async def main():
    # 执行全量爬取（适合首次运行或爬取指定时间段）
    # 如果只想更新，可以使用 scraper.fetch_update()，但通常 fetch_all 更快
    print("正在执行 fetch_all...")
    await scraper.fetch_all()
    print("爬取完成.")

    # 将结果输出到 Markdown 文件
    # 文件名会根据日期自动生成，例如 output_llms/YYYY-MM-DD.md
    # meta=True 会在文件开头添加爬取元信息
    print("正在生成 Markdown 文件...")
    scraper.to_markdown(meta=True)
    print("Markdown 文件已生成.")

    # (可选) 输出到 CSV 文件，方便导入飞书等
    # scraper.to_csv(csv_config=dict(delimiter="\t", header=False))
    # print("CSV 文件已生成.")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
    print("爬取任务结束.") 