#!/usr/bin/env python3
"""
Test the STEM Judge and Web Summary services that run on remote servers.

Updates:
- Force the service addresses to be configured through environment variables (no default fallback):
  - STEM: STEM_LLM_JUDGE_URL (e.g. http://localhost:8004 or a base that already includes /v1)
  - Web Summary: WEB_SUMMARY_API_BASE or EB_SUMMARY_API_BASE (e.g. http://localhost:8080/v1)
- Automatically derive the /health and /v1/chat/completions endpoints based on whether the base already ends with /v1.
- Keep the model identifiers fixed to the user-provided IDs without probing or falling back:
  - Web Summary: Qwen/Qwen3-32B
  - STEM Judge: openai/gpt-oss-20b
"""

import requests
import json
import time
from datetime import datetime
import os
from urllib.parse import urlparse, urlunparse
import re

# Fixed model identifiers provided by the user
WEB_SUMMARY_MODEL_NAME = "Qwen/Qwen3-32B"  # Updated to the actually available model
STEM_MODEL_NAME = "openai/gpt-oss-20b"

# Read service base URLs from environment variables without falling back to defaults
_stem_env = os.getenv("STEM_LLM_JUDGE_URL")
STEM_BASE = _stem_env.rstrip("/") if _stem_env else None

_web_env = os.getenv("WEB_SUMMARY_API_BASE", os.getenv("EB_SUMMARY_API_BASE"))
WEB_SUMMARY_BASE = _web_env.rstrip("/") if _web_env else None

# When environment variables are not set, try the following candidates automatically
WEB_SUMMARY_CANDIDATES = []  # Provide candidates via env vars if discovery is desired

STEM_CANDIDATES = []  # Provide candidates via env vars if discovery is desired


def _strip_v1_from_path(path: str) -> str:
    """Remove a trailing /v1 or /v1/ fragment from the path."""
    return re.sub(r"/v1/?$", "", path.rstrip("/")) or "/"


def build_health_url(base_url: str) -> str:
    """Build the health endpoint, replacing a trailing /v1 with the root."""
    parsed = urlparse(base_url)
    base_path = parsed.path or "/"
    root_path = _strip_v1_from_path(base_path)
    # Ensure there is only a single slash
    health_path = (root_path.rstrip("/") + "/health") if root_path != "/" else "/health"
    return urlunparse((parsed.scheme, parsed.netloc, health_path, "", "", ""))


def build_completions_url(base_url: str) -> str:
    """Build the /v1/chat/completions endpoint without duplicating /v1."""
    parsed = urlparse(base_url)
    base_path = parsed.path or ""
    if re.search(r"/v1/?$", base_path):
        path = base_path.rstrip("/") + "/chat/completions"
    else:
        path = base_path.rstrip("/") + "/v1/chat/completions"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def test_health_endpoint(service_name, url):
    """Exercise a service health-check endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing {service_name} health endpoint")
    print(f"URL: {url}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            print(f"✅ Status: OK (HTTP {response.status_code})")
            print(f"Response body: {response.text[:200]}")
            return True
        else:
            print(f"⚠️ Status: Unexpected (HTTP {response.status_code})")
            print(f"Response body: {response.text[:200]}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {str(e)}")
        return False


def try_health_on_base(base_url: str, timeout: int = 3) -> bool:
    """Try calling /health for the given base URL."""
    url = build_health_url(base_url)
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def discover_base(service_name: str, candidates: list) -> str:
    """Probe the candidate list and return the first healthy base URL, if any."""
    print(f"\nStarting {service_name} base URL discovery...")
    for i, base in enumerate(candidates, 1):
        health_url = build_health_url(base)
        print(f"  [{i:02d}] Trying: {health_url}", end=" ")
        ok = try_health_on_base(base)
        if ok:
            print("=> ✅ Available")
            return base.rstrip("/")
        else:
            print("=> ❌ Unavailable")
    print(f"Unable to discover a reachable {service_name} service")
    return None


def test_stem_judge_inference():
    """Run inference checks against the STEM Judge service."""
    print(f"\n{'='*60}")
    print("Testing STEM Judge inference")
    print("-" * 60)

    if not STEM_BASE:
        # Auto-discover when the base URL is not preset
        base = discover_base("STEM Judge", STEM_CANDIDATES)
        if not base:
            print("❌ Unable to discover a STEM service, skipping inference tests")
            return
        globals()["STEM_BASE"] = base

    url = build_completions_url(STEM_BASE)

    # Example test cases
    test_cases = [
        {
            "name": "Simple arithmetic (correct answer)",
            "question": "1+1=?",
            "ground_truth": "2",
            "student_answer": "2",
            "expected": "Yes",
        },
        {
            "name": "Simple arithmetic (incorrect answer)",
            "question": "1+1=?",
            "ground_truth": "2",
            "student_answer": "3",
            "expected": "No",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {test_case['name']}")
        print("-" * 40)

        # Build the judge prompt
        prompt = f"""### Question: {test_case['question']}

### Ground Truth Answer: {test_case['ground_truth']}

### Student Answer: {test_case['student_answer']}

If correct, output "Final Decision: Yes" else "Final Decision: No"."""

        # Build the OpenAI-compatible request payload
        payload = {
            "model": STEM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 100,
        }

        print(f"Sending request to: {url}")
        print(f"Expected result: Final Decision: {test_case['expected']}")

        try:
            start_time = time.time()
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    print(f"✅ Inference succeeded (elapsed: {elapsed_time:.2f} seconds)")
                    print(f"Model response: {content}")

                    # Verify the response matches expectations
                    if f"Final Decision: {test_case['expected']}" in content:
                        print("✅ Result matches expectation")
                    else:
                        print("⚠️ Result may not match expectation")
                else:
                    print("⚠️ Response format is unexpected")
                    print(f"Response: {json.dumps(result, indent=2)}")
            else:
                print(f"❌ Request failed (HTTP {response.status_code})")
                print(f"Error details: {response.text[:500]}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {str(e)}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {str(e)}")
            print(f"Raw response: {response.text[:500]}")


def test_web_summary():
    """Run inference checks against the Web Summary service."""
    print(f"\n{'='*60}")
    print("Testing Web Summary service")
    print("-" * 60)

    if not WEB_SUMMARY_BASE:
        # Auto-discover when the base URL is not preset
        base = discover_base("Web Summary", WEB_SUMMARY_CANDIDATES)
        if not base:
            print("❌ Unable to discover a Web Summary service, skipping inference tests")
            return
        globals()["WEB_SUMMARY_BASE"] = base

    # Start by testing the health endpoint
    health_url = build_health_url(WEB_SUMMARY_BASE)
    if test_health_endpoint("Web Summary", health_url):
        print("\nRunning Web Summary inference checks...")

        # Then exercise the inference endpoint
        url = build_completions_url(WEB_SUMMARY_BASE)

        test_content = """
        This is a test article about artificial intelligence.
        AI has made significant progress in recent years.
        Machine learning models can now perform various tasks.
        """

        # Always use the specified model name
        payload = {
            "model": WEB_SUMMARY_MODEL_NAME,
            "messages": [{"role": "user", "content": f"Please summarize the following text:\n\n{test_content}"}],
            "temperature": 0.7,
            "max_tokens": 150,
        }

        print(f"Sending test request to: {url}")
        print(f"Using model: {WEB_SUMMARY_MODEL_NAME}")

        try:
            start_time = time.time()
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    print(f"✅ Inference succeeded (elapsed: {elapsed_time:.2f} seconds)")
                    print(f"Summary result: {content[:500]}")
                else:
                    print("⚠️ Response format is unexpected")
                    print(f"Response: {json.dumps(result, indent=2)[:500]}")
            else:
                print(f"⚠️ Inference endpoint might be unavailable (HTTP {response.status_code})")
                print(f"Response: {response.text[:500]}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {str(e)}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {str(e)}")


def main():
    """Entry point for ad-hoc remote service validation."""
    global STEM_BASE, WEB_SUMMARY_BASE
    print("=" * 60)
    print("Remote service testing utility")
    print(f"STEM base URL: {STEM_BASE or 'Not configured (will discover automatically)'}")
    print(f"Web Summary base URL: {WEB_SUMMARY_BASE or 'Not configured (will discover automatically)'}")
    print("=" * 60)

    # Discover base URLs when they are not pre-configured
    if not WEB_SUMMARY_BASE:
        WEB_SUMMARY_BASE = discover_base("Web Summary", WEB_SUMMARY_CANDIDATES)
    if not STEM_BASE:
        STEM_BASE = discover_base("STEM Judge", STEM_CANDIDATES)

    # Test health-check endpoints
    web_summary_health = False
    stem_judge_health = False

    if WEB_SUMMARY_BASE:
        print(f"  Web Summary health URL: {build_health_url(WEB_SUMMARY_BASE)}")
        web_summary_health = test_health_endpoint("Web Summary", build_health_url(WEB_SUMMARY_BASE))
    if STEM_BASE:
        print(f"  STEM Judge health URL: {build_health_url(STEM_BASE)}")
        stem_judge_health = test_health_endpoint("STEM Judge", build_health_url(STEM_BASE))

    # Run inference tests when the health-check passes
    if stem_judge_health:
        test_stem_judge_inference()
    else:
        if STEM_BASE:
            print("\n⚠️ Skipping STEM Judge inference test (health check failed)")

    if web_summary_health:
        test_web_summary()
    else:
        if WEB_SUMMARY_BASE:
            print("\n⚠️ Skipping Web Summary inference test (health check failed)")

    # Summarize the checks at the end
    print(f"\n{'='*60}")
    print("Test summary")
    print("-" * 60)
    print(f"Web Summary health check: {'✅ Passed' if web_summary_health else '❌ Failed or skipped'}")
    print(f"STEM Judge health check: {'✅ Passed' if stem_judge_health else '❌ Failed or skipped'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
