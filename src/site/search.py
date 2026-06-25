import asyncio
from pathlib import Path

from pagefind.index import PagefindIndex, IndexConfig


def build_search_index(html_dir: Path) -> None:
    asyncio.run(_build(html_dir))


async def _build(html_dir: Path) -> None:
    config = IndexConfig(output_path="_pagefind")
    async with PagefindIndex(config=config) as index:
        await index.add_directory(str(html_dir), glob="vocab/*.{html}")
        await index.write_files(output_path=str(html_dir / "_pagefind"))
