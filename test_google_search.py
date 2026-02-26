import os
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

async def run():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Optional: ensure we test with cache if possible, but for now just basic check
    print("Testing if tool works without cache...")
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.3
    )
    res = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents="what's the weather in NY?",
        config=config
    )
    print("Generated text config:\n", res.text[:200])
    
    try:
        from cache_manager import get_cache_name
        c_name = get_cache_name()
        if c_name:
            print(f"Testing with Cache {c_name} ...")
            config2 = types.GenerateContentConfig(
                cached_content=c_name,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3
            )
            # CACHE Requires that system_instruction is empty and contexts is mapped to user part
            res2 = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents="what's the weather in NY today? answer briefly.",
                config=config2
            )
            print("Generated text with CACHE:\n", res2.text[:200])
        else:
            print("No cache configured for tests.")
    except Exception as e:
        print("Cache err:", e)
    
asyncio.run(run())
