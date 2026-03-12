#!/usr/bin/env python3
"""Kill all Daytona sandboxes. Run between training iterations to clean up orphaned sandboxes."""

import asyncio
from daytona import AsyncDaytona


async def main():
    async with AsyncDaytona() as daytona:
        page = await daytona.list()
        sandboxes = page.items or []
        if not sandboxes:
            print("No sandboxes found.")
            return
        print(f"Found {len(sandboxes)} sandbox(es) (page 1/{page.total_pages}). Deleting...")
        deleted = 0
        for sb in sandboxes:
            try:
                await daytona.delete(sb)
                deleted += 1
            except Exception as e:
                print(f"  Failed to delete sandbox {sb.id}: {e}")
        # Handle additional pages
        for p in range(2, (page.total_pages or 1) + 1):
            next_page = await daytona.list()
            for sb in (next_page.items or []):
                try:
                    await daytona.delete(sb)
                    deleted += 1
                except Exception as e:
                    print(f"  Failed to delete sandbox {sb.id}: {e}")
        print(f"Done. Deleted {deleted} sandbox(es).")


if __name__ == "__main__":
    asyncio.run(main())
