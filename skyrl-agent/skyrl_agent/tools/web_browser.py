from skyrl_agent.tools.base import BaseTool, register_tool
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
import requests
from skyrl_agent.tools.prompt import EXTRACTOR_PROMPT
import os
from openai import OpenAI
import random
from skyrl_agent.tools.cache import WebPageCache
import copy


@register_tool("web_browser")
class WebBrowser(BaseTool):
    name = "web_browser"
    description = (
        "Visit webpage(s) and return the summary of the content.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "- Search results alone are NOT sufficient - you need full webpage content\n"
        "- If search returns papers/PDFs, you MUST visit them with web_browser to extract key findings\n"
        "- Skipping web_browser will result in incomplete and unreliable answers"
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL string or an array of URL strings. Use plain URLs only, never Markdown format.",
            },
            "goal": {"type": "string", "description": "The goal of the visit for webpage(s)."},
        },
        "required": ["url", "goal"],
    }

    webcontent_maxlength = int(os.getenv("WEBCONTENT_MAXLENGTH", "150000"))
    ignore_jina = os.getenv("IGNORE_JINA", "false").lower() == "true"
    jina_reader_url_prefix = "https://r.jina.ai/"
    jina_api_keys = os.getenv("JINA_API_KEYS", "").split(",") if os.getenv("JINA_API_KEYS") else []
    use_jina = len(jina_api_keys) > 0  # Only use Jina if we have API keys
    # Timeouts and retries (tunable via env)
    _default_max_attempts = int(os.getenv("BROWSER_MAX_ATTEMPTS", "2"))
    _jina_timeout_s = float(os.getenv("JINA_TIMEOUT_SECONDS", "5"))
    _jina_retries = int(os.getenv("JINA_MAX_RETRIES", "2"))
    # ASearcher-style length controls (only affect content size and chunking)
    page_char_cap = int(os.getenv("BROWSER_PAGE_MAX_CHARS", "240000"))
    chunk_size = int(os.getenv("BROWSER_CHUNK_SIZE", "16000"))
    max_chunks = int(os.getenv("BROWSER_MAX_CHUNKS", "15"))
    # Shared persistent page cache (ASearcher-style)
    _page_cache = WebPageCache(
        max_size=10000,
        cache_file=os.getenv(
            "WEBPAGE_CACHE_FILE",
            os.path.join(
                os.getenv("SKYAGENT_WEB_CACHE_DIR", os.path.expanduser("~/.skyagent_web_cache")),
                "webpage_cache.json",
            ),
        ),
    )

    # Optional domain/keyword blocklists to avoid benchmark leakage during eval
    _block_domains_env = os.getenv("WEB_BLOCKLIST_DOMAINS", os.getenv("SEARCH_BLOCKLIST_DOMAINS", "")).strip()
    _block_keywords_env = os.getenv("WEB_BLOCKLIST_KEYWORDS", os.getenv("SEARCH_BLOCKLIST_KEYWORDS", "")).strip()
    block_domains = {d.strip().lower() for d in _block_domains_env.split(",") if d.strip()}
    block_keywords = {k.strip().lower() for k in _block_keywords_env.split(",") if k.strip()}

    def _is_blocked_url(self, url: str) -> bool:
        try:
            lu = (url or "").lower()
            if self.block_domains and any(d in lu for d in self.block_domains):
                return True
            if self.block_keywords and any(k in lu for k in self.block_keywords):
                return True
        except Exception:
            return False
        return False

    def _normalize_url(self, url: str) -> str:
        """Minimal normalization for anti-bot wrappers and PDF endpoints.

        - Unwrap validate.perfdrive.com links (decode 'ssc' target).
        - For iopscience.iop.org article PDFs/EPUBs (path ends with '/pdf' or '/epub'),
          prefer the HTML article page by dropping the suffix.
        - Drop leading 'www.' when host already has multiple labels (e.g., www.pmc.ncbi.nlm.nih.gov → pmc.ncbi.nlm.nih.gov).
        """
        try:
            from urllib.parse import urlparse, parse_qs, unquote, urlunparse

            u = (url or "").strip()
            if not u:
                return url
            p = urlparse(u)

            # Unwrap Radware/Perfdrive wrapper
            if p.netloc.endswith("validate.perfdrive.com"):
                qs = parse_qs(p.query)
                ssc = qs.get("ssc", [""])[0]
                if ssc:
                    target = unquote(ssc)
                    return self._normalize_url(target)

            # Prefer HTML article page over direct PDF/EPUB on iopscience
            if p.netloc.endswith("iopscience.iop.org") and (p.path.endswith("/pdf") or p.path.endswith("/epub")):
                new_path = p.path.rsplit("/", 1)[0]
                return urlunparse((p.scheme, p.netloc, new_path, p.params, "", ""))

            # Drop www. for multi-label hosts (general fix for www.{sub}.{domain})
            host = p.netloc
            if host.startswith("www.") and host.count(".") >= 2:
                host = host[4:]
                return urlunparse((p.scheme, host, p.path, p.params, p.query, p.fragment))

            return u
        except Exception:
            return url

    def _url_variants(self, url: str) -> List[str]:
        """Generate generic URL variants to handle common fetch failures.

        Variants include toggling scheme (https/http) and adding/removing a
        leading 'www.' for simple two-label hosts, or removing it for
        multi-label hosts. This is general and not site-specific.
        """
        try:
            from urllib.parse import urlparse, urlunparse

            base = (url or "").strip()
            if not base:
                return []
            p = urlparse(base)
            # Schemes to try (prefer https first)
            scheme0 = p.scheme if p.scheme else "https"
            schemes = [scheme0, ("http" if scheme0 == "https" else "https")]
            # Host variants
            hosts = []
            host = p.netloc
            if host:
                hosts.append(host)
                if host.startswith("www."):
                    # For multi-label hosts like www.sub.example.com, prefer dropping www.
                    hosts.append(host[4:])
                else:
                    # Only for simple two-label hosts like example.com, try adding www.
                    if host.count(".") == 1:
                        hosts.append("www." + host)

            variants = []
            for sc in schemes:
                for h in hosts:
                    cand = urlunparse((sc, h, p.path, p.params, p.query, p.fragment))
                    variants.append(cand)
            # Deduplicate while preserving order
            seen = set()
            uniq = []
            for v in variants:
                if v and v not in seen:
                    seen.add(v)
                    uniq.append(v)
            return uniq or [base]
        except Exception:
            return [url]

    def call(self, params: dict, **kwargs) -> Union[str, dict]:
        """
        Visits webpage(s) and returns summarized content.

        Args:
            params (dict): Dictionary containing 'url' and 'goal'.
            **kwargs: Additional keyword arguments.

        Returns:
            str or dict: The webpage summary or an error message.
        """
        # Verify required parameters (be tolerant of schema/JSON issues)
        # Capture trajectory id for sticky routing (optional)
        try:
            self._trajectory_id = kwargs.get("trajectory_id")
        except Exception:
            self._trajectory_id = None

        try:
            params = self._verify_json_format_args(params)
        except Exception as e:
            return {
                "error": f"Invalid parameters: {str(e)}",
                "hint": "Pass plain URL string or an array of URL strings in 'url', and a non-empty 'goal' string.",
            }

        try:
            url = params["url"]
            goal = params["goal"]
        except KeyError as e:
            return {"error": f"Missing required field: {str(e)}"}

        if not goal:
            return {
                "error": "Goal parameter is required.",
                "hint": "Provide a concise description of what to extract from the page.",
            }

        try:
            # Handle case where url is a JSON string that looks like a list
            if isinstance(url, str):
                # Strip extra quotes if present
                url = url.strip()
                if (url.startswith('"') and url.endswith('"')) or (url.startswith("'") and url.endswith("'")):
                    url = url[1:-1]
                    print(f"Removed extra quotes from URL: {url}")

                # First, check if it's a Markdown link format [text](url)
                import re

                markdown_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
                markdown_match = re.match(markdown_pattern, url.strip())
                if markdown_match:
                    # Extract the actual URL from Markdown format
                    url = markdown_match.group(2)
                    print(f"Extracted URL from Markdown format: {url}")

                # Check if it's a JSON array string
                url_stripped = url.strip()
                if url_stripped.startswith("[") and url_stripped.endswith("]"):
                    try:
                        parsed = json.loads(url_stripped)
                        if isinstance(parsed, list):
                            # Clean any Markdown formatted URLs in the list
                            cleaned_urls = []
                            for u in parsed:
                                if isinstance(u, str):
                                    # Check if this URL is also in Markdown format
                                    md_match = re.match(markdown_pattern, u.strip())
                                    if md_match:
                                        cleaned_urls.append(md_match.group(2))
                                    else:
                                        cleaned_urls.append(u)
                            if cleaned_urls:
                                url = cleaned_urls
                                # print(f"Parsed URL list: {url}")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON array: {e}")
                        # Try to extract URLs from malformed array like [url1, url2]
                        inner_content = url_stripped[1:-1]  # Remove brackets
                        # Split by comma and clean each URL
                        potential_urls = inner_content.split(",")
                        cleaned_urls = []
                        for u in potential_urls:
                            u = u.strip()
                            # Remove any quotes if present
                            if (u.startswith('"') and u.endswith('"')) or (u.startswith("'") and u.endswith("'")):
                                u = u[1:-1]
                            # Check for Markdown format
                            md_match = re.match(markdown_pattern, u)
                            if md_match:
                                cleaned_urls.append(md_match.group(2))
                            elif u.startswith("http://") or u.startswith("https://"):
                                cleaned_urls.append(u)
                        if cleaned_urls:
                            url = cleaned_urls
                            print(f"Extracted URLs from malformed array: {url}")
                        # Otherwise keep as string

                # Fallbacks for stray leading '[' or embedded URLs within a string
                if isinstance(url, str):
                    # Extract any http(s) URLs embedded in the string
                    candidates = re.findall(r'https?://[^\s\]\\)"\'<>]+', url)
                    if candidates:
                        # Deduplicate while preserving order
                        seen = set()
                        dedup = []
                        for c in candidates:
                            if c not in seen:
                                seen.add(c)
                                dedup.append(c)
                        url = dedup[0] if len(dedup) == 1 else dedup
                        print(f"Extracted URL(s) from freeform string: {url}")
                    else:
                        # If starts with stray '[' but no closing ']', strip and retry extraction once
                        if url_stripped.startswith("[") and not url_stripped.endswith("]"):
                            candidate = url_stripped.lstrip("[").rstrip("]")
                            candidates2 = re.findall(r'https?://[^\s\]\\)"\'<>]+', candidate)
                            if candidates2:
                                url = candidates2[0] if len(candidates2) == 1 else candidates2
                                print(f"Recovered URL(s) from stray-bracket string: {url}")

            if isinstance(url, str):
                url = self._normalize_url(url)
                if self._is_blocked_url(url):
                    response = (
                        f"The useful information in {url} for user goal {goal} as follows: \n\n"
                        "Evidence in page: \nBlocked by evaluation policy (domain/keyword blocklist).\n\n"
                        "Summary: \nThis URL is excluded to avoid benchmark leakage.\n\n"
                    )
                else:
                    response = self.readpage(url, goal)
            elif isinstance(url, list):
                response = []
                with ThreadPoolExecutor(max_workers=3) as executor:
                    # Skip blocked URLs
                    urls_to_fetch = []
                    for u in url:
                        if isinstance(u, str):
                            u = self._normalize_url(u)
                        if self._is_blocked_url(u):
                            response.append(
                                f"The useful information in {u} for user goal {goal} as follows: \n\n"
                                "Evidence in page: \nBlocked by evaluation policy (domain/keyword blocklist).\n\n"
                                "Summary: \nThis URL is excluded to avoid benchmark leakage.\n\n"
                            )
                        else:
                            urls_to_fetch.append(u)
                    futures = {executor.submit(self.readpage, u, goal): u for u in urls_to_fetch}
                    for future in as_completed(futures):
                        try:
                            response.append(future.result())
                        except Exception as e:
                            response.append(f"Error fetching {futures[future]}: {str(e)}")
                response = "\n=======\n".join(response)
            else:
                return {"error": "URL must be a string or array of strings."}

            # print(f'Summary Length {len(response)}; Summary Content {response}')
            return {"results": response.strip()}

        except Exception as e:
            return {"error": f"Web browsing failed: {str(e)}"}

    def call_server(self, msgs, max_tries=10):
        """
        Call the OpenAI API server to process webpage content.

        Args:
            msgs: Messages to send to the API
            max_tries: Maximum number of retry attempts

        Returns:
            str: The API response content
        """
        openai_api_key = "EMPTY"
        openai_api_base = os.getenv("WEB_SUMMARY_API_BASE")
        summary_model = os.getenv("WEB_SUMMARY_MODEL")
        assert openai_api_base, "WEB_SUMMARY_API_BASE is not set"
        assert summary_model, "WEB_SUMMARY_MODEL is not set"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        def _shrink_extractor_prompt_content(text: str, ratio: float = 0.5) -> str:
            """Shrink only the Webpage Content section inside EXTRACTOR_PROMPT.

            Falls back to trimming the whole message if markers are not found.
            """
            head_marker = "## **Webpage Content**"
            goal_marker = "## **User Goal**"
            try:
                head_idx = text.find(head_marker)
                goal_idx = text.find(goal_marker)
                if head_idx != -1 and goal_idx != -1 and goal_idx > head_idx:
                    # Start of the variable content is the newline after the header
                    nl_idx = text.find("\n", head_idx + len(head_marker))
                    content_start = nl_idx + 1 if nl_idx != -1 else head_idx + len(head_marker)
                    # Existing content segment to shrink
                    segment = text[content_start:goal_idx]
                    keep = max(1, int(len(segment) * ratio))
                    return text[:content_start] + segment[:keep] + text[goal_idx:]
            except Exception:
                pass
            # Fallback: trim entire text
            return text[: max(1, int(len(text) * ratio))]

        def _shrink_messages(msgs_in: list, ratio: float = 0.5) -> list:
            new_msgs = copy.deepcopy(msgs_in)
            for m in new_msgs:
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                    m["content"] = _shrink_extractor_prompt_content(m["content"], ratio)
            return new_msgs

        def _split_extractor_prompt_content(text: str) -> tuple[str, str]:
            head_marker = "## **Webpage Content**"
            goal_marker = "## **User Goal**"
            try:
                head_idx = text.find(head_marker)
                goal_idx = text.find(goal_marker)
                if head_idx != -1 and goal_idx != -1 and goal_idx > head_idx:
                    nl_idx = text.find("\n", head_idx + len(head_marker))
                    content_start = nl_idx + 1 if nl_idx != -1 else head_idx + len(head_marker)
                    segment = text[content_start:goal_idx]
                    half = len(segment) // 2
                    first = text[:content_start] + segment[:half] + text[goal_idx:]
                    second = text[:content_start] + segment[half:] + text[goal_idx:]
                    return first, second
            except Exception:
                pass
            mid = max(1, len(text) // 2)
            return text[:mid], text[mid:]

        def _split_messages_into_two(msgs_in: list) -> tuple[list, list]:
            a = copy.deepcopy(msgs_in)
            b = copy.deepcopy(msgs_in)
            # find largest user content
            idx = -1
            best = -1
            for k, m in enumerate(a):
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                    L = len(m["content"])
                    if L > best:
                        best = L
                        idx = k
            if idx >= 0:
                first, second = _split_extractor_prompt_content(a[idx]["content"])
                a[idx]["content"] = first
                b[idx]["content"] = second
            return a, b

        def _normalize_json_str(s: str) -> str:
            if not s:
                return s
            try:
                json.loads(s)
                return s
            except Exception:
                left = s.find("{")
                right = s.rfind("}")
                if left != -1 and right != -1 and left <= right:
                    return s[left : right + 1]
                return s

        def _combine_json_strs(a: str, b: str) -> str:
            a = _normalize_json_str(a)
            b = _normalize_json_str(b)
            try:
                da = json.loads(a) if a else {}
            except Exception:
                da = {}
            try:
                db = json.loads(b) if b else {}
            except Exception:
                db = {}
            if not da and not db:
                return ""
            rational = (da.get("rational") or "").strip()
            if not rational:
                rational = (db.get("rational") or "").strip()
            ev = "\n\n".join(
                [x for x in [str(da.get("evidence", "")).strip(), str(db.get("evidence", "")).strip()] if x]
            )
            su = "\n\n".join([x for x in [str(da.get("summary", "")).strip(), str(db.get("summary", "")).strip()] if x])
            try:
                return json.dumps({"rational": rational, "evidence": ev, "summary": su}, ensure_ascii=False)
            except Exception:
                return (a or "") + ("\n\n" if a and b else "") + (b or "")

        def _call_once_with_shrink(msgs_try: list) -> str:
            local_msgs = copy.deepcopy(msgs_try)
            local_shrink = 2
            while True:
                try:
                    resp = client.chat.completions.create(
                        model=summary_model,
                        messages=local_msgs,
                        stop=["\n<tool_response>", "<tool_response>"],
                        temperature=0,
                        user=(str(self._trajectory_id) if getattr(self, "_trajectory_id", None) else None),
                    )
                    cont = resp.choices[0].message.content
                    return _normalize_json_str(cont or "")
                except Exception as ie:
                    ses = str(ie).lower()
                    ictx = (
                        "maximum context length" in ses
                        or ("context" in ses and "exceed" in ses)
                        or "reduce the length" in ses
                        or "too many tokens" in ses
                    )
                    if ictx and local_shrink > 0:
                        local_msgs = _shrink_messages(local_msgs, 0.5)
                        local_shrink -= 1
                        continue
                    return ""

        # Work on a local copy we can shrink progressively on context errors
        msgs_local = copy.deepcopy(msgs)
        shrink_attempts = 3

        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=summary_model,
                    messages=msgs_local,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=0,
                    user=(str(self._trajectory_id) if getattr(self, "_trajectory_id", None) else None),
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except Exception:
                        # extract json from string
                        left = content.find("{")
                        right = content.rfind("}")
                        if left != -1 and right != -1 and left <= right:
                            content = content[left : right + 1]
                    return content
            except Exception as e:
                es = str(e).lower()
                ctx_err = (
                    "maximum context length" in es
                    or "context length" in es
                    or "reduce the length" in es
                    or "too many tokens" in es
                    or "exceed" in es
                    and "context" in es
                )
                if ctx_err:
                    a_msgs, b_msgs = _split_messages_into_two(msgs_local)
                    part_a = _call_once_with_shrink(a_msgs)
                    part_b = _call_once_with_shrink(b_msgs)
                    combined = _combine_json_strs(part_a, part_b)
                    if combined:
                        return combined
                    # fallback to shrinking the original message if split failed
                    if shrink_attempts > 0:
                        msgs_local = _shrink_messages(msgs_local, 0.5)
                        shrink_attempts -= 1
                        continue
                if attempt == (max_tries - 1):
                    return ""
                continue
        return ""

    def jina_readpage(self, url: str) -> str:
        """Read webpage using Jina (only if API keys available)"""
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            
        Returns:
            str: The webpage content or error message
        """
        if not self.jina_api_keys:
            return "[visit] No Jina API keys available."

        headers = {
            "Authorization": f"Bearer {random.choice(self.jina_api_keys)}",
        }
        # Keep retries/timeouts conservative to reduce wall time; tunable via env
        max_retries = max(1, int(getattr(self, "_jina_retries", 2)))
        timeout = max(1.0, float(getattr(self, "_jina_timeout_s", 5.0)))

        # Try generic URL variants (scheme/www) within each retry attempt
        for attempt in range(max_retries):
            try:
                for target in self._url_variants(url) or [url]:
                    try:
                        response = requests.get(f"https://r.jina.ai/{target}", headers=headers, timeout=timeout)
                        if response.status_code == 200 and response.text:
                            return response.text
                        else:
                            # Print remote error body as before for visibility
                            try:
                                print(response.text)
                            except Exception:
                                pass
                            # Raise to align with original retry flow
                            raise ValueError("jina readpage error")
                    except Exception:
                        # Try next variant
                        continue
            except Exception:
                # Fall through to next attempt
                pass
            if attempt == max_retries - 1:
                return "[visit] Failed to read page."

        return "[visit] Failed to read page."

    def readpage(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content and extract relevant information.

        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page

        Returns:
            str: The processed webpage content or error message
        """
        # Normalize upfront to avoid anti-bot wrappers and blocked PDFs
        url = self._normalize_url(url)

        # Avoid repeated hammering of the same failing URL within a short TTL
        try:
            if not hasattr(self, "_recent_failures"):
                self._recent_failures = {}
            ttl = int(os.getenv("WEB_BROWSER_FAIL_TTL", "10"))
        except Exception:
            ttl = 10
        now = None
        try:
            now = __import__("time").time()
            last = self._recent_failures.get(url)
            if last and (now - last) < ttl:
                # Still print a concise note to signal skip
                print(f"[web_browser][skip] recently failed url={url} since={int(now-last)}s ttl={ttl}s")
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal
                )
                useful_information += (
                    "Evidence in page: \n"
                    + "Recently attempted but unreachable; skipping re-access to avoid consecutive failures."
                    + "\n\n"
                )
                useful_information += (
                    "Summary: \n"
                    + "Prior attempts failed very recently. Try alternative sources or retry later."
                    + "\n\n"
                )
                return useful_information
        except Exception:
            pass

        # Keep overall attempts small to avoid long waits (env: BROWSER_MAX_ATTEMPTS)
        try:
            max_attempts = max(1, int(self._default_max_attempts))
        except Exception:
            max_attempts = 2
        for attempt in range(max_attempts):
            # Try cache first (exactly like ASearcher-style persistent LRU)
            content = self._page_cache.get(url)
            service = "cache" if content else "jina"
            if not content:
                content = self.jina_readpage(url)

            # print(service)
            # print(content)
            if (
                content
                and not content.startswith("[visit] Failed to read page.")
                and content != "[visit] Empty content."
                and not content.startswith("[document_parser]")
            ):
                # On valid fetch (including first-time), populate cache
                if service == "jina":
                    try:
                        self._page_cache.put(url, content)
                    except Exception:
                        pass
                # ASearcher-style length handling: cap page size and split into chunks, summarize per chunk
                content = content[: self.page_char_cap]

                # Build chunks: 10k each, up to 10 chunks
                chunks = []
                lim = min(len(content), self.chunk_size * self.max_chunks)
                i = 0
                while i < lim and len(chunks) < self.max_chunks:
                    chunks.append(content[i : i + self.chunk_size])
                    i += self.chunk_size

                aggregated_evidence = []
                aggregated_summary = []

                for chunk in chunks:
                    messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=chunk, goal=goal)}]
                    raw = self.call_server(messages)

                    # Handle long chunk content by progressive truncation
                    summary_retries = 3
                    current = chunk
                    while (not raw or len(raw) < 10) and summary_retries >= 0:
                        truncate_length = int(0.7 * len(current)) if summary_retries > 0 else min(25000, len(current))
                        current = current[:truncate_length]
                        extraction_prompt = EXTRACTOR_PROMPT.format(webpage_content=current, goal=goal)
                        messages = [{"role": "user", "content": extraction_prompt}]
                        raw = self.call_server(messages)
                        summary_retries -= 1

                    # Parse JSON response with a few retries
                    parse_retry_times = 0
                    parsed = None
                    while parse_retry_times < 3:
                        try:
                            parsed = json.loads(raw)
                            break
                        except Exception:
                            raw = self.call_server(messages)
                            parse_retry_times += 1

                    if parsed and isinstance(parsed, dict):
                        aggregated_evidence.append(str(parsed.get("evidence", "")))
                        aggregated_summary.append(str(parsed.get("summary", "")))

                # Generate final aggregated response if any chunk succeeded
                if aggregated_evidence or aggregated_summary:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                        url=url, goal=goal
                    )
                    useful_information += "Evidence in page: \n" + "\n\n".join(aggregated_evidence) + "\n\n"
                    useful_information += "Summary: \n" + "\n\n".join(aggregated_summary) + "\n\n"
                    return useful_information

                # If all chunks failed to parse, fall back to original failure message
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal
                )
                useful_information += (
                    "Evidence in page: \n"
                    + "The provided webpage content could not be accessed. Please check the URL or file format."
                    + "\n\n"
                )
                useful_information += (
                    "Summary: \n"
                    + "The webpage content could not be processed, and therefore, no information is available."
                    + "\n\n"
                )
                return useful_information

            # If we're on the last attempt, record failure and return failure message
            if attempt == max_attempts - 1:
                try:
                    if now is None:
                        now = __import__("time").time()
                    self._recent_failures[url] = now
                except Exception:
                    pass
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal
                )
                useful_information += (
                    "Evidence in page: \n"
                    + "The provided webpage content could not be accessed. Please check the URL or file format."
                    + "\n\n"
                )
                useful_information += (
                    "Summary: \n"
                    + "The webpage content could not be processed, and therefore, no information is available."
                    + "\n\n"
                )
                return useful_information


if __name__ == "__main__":
    # Example usage for testing
    tool = WebBrowser()
    test_params = {"url": "https://apple.com", "goal": "Find information about the company"}
    result = tool.call(test_params)
    print("Test Result:", result)
