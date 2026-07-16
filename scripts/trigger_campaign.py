import sys
import httpx
import argparse

def main():
    parser = argparse.ArgumentParser(description="Trigger the Jurexia Subscriber Campaign via Vercel Production API")
    parser.add_argument("--test", type=str, help="Send a test email to the specified address (e.g. jdm.juridico@gmail.com)")
    parser.add_argument("--send", action="store_true", help="Execute the actual campaign (dryRun=False)")
    parser.add_argument("--limit", type=int, default=10, help="Batch limit (default: 10)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N users (default: 0)")
    
    args = parser.parse_args()
    
    url = "https://www.iurexia.com/api/email-campaign?key=jurexia-reingest-2026"
    
    payload = {}
    if args.test:
        payload["testEmails"] = [args.test]
        print(f"Triggering test email to {args.test}...")
    elif args.send:
        payload["dryRun"] = False
        payload["limit"] = args.limit
        payload["offset"] = args.offset
        print(f"Triggering ACTUAL campaign (dryRun=False, limit={args.limit}, offset={args.offset})...")
    else:
        payload["dryRun"] = True
        payload["limit"] = args.limit
        payload["offset"] = args.offset
        print(f"Triggering DRY RUN campaign (limit={args.limit}, offset={args.offset})...")
        
    try:
        resp = httpx.post(url, json=payload, timeout=60.0)
        if resp.status_code == 200:
            print("Success!")
            import json
            print(json.dumps(resp.json(), indent=2))
        else:
            print(f"Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()
