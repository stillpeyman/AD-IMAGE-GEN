import asyncio
import time

async def slow_operation(name):
    print(f"{name} starting...")
    await asyncio.sleep(2)  # Simulates API call
    print(f"{name} finished!")

async def main():
    # These run concurrently (not one after the other)
    await asyncio.gather(
        slow_operation("User 1"),
        slow_operation("User 2"), 
        slow_operation("User 3")
    )

# Run this to see async in action:
asyncio.run(main())