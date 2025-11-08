import os
import requests
import time
import random
from typing import Optional, Union, Dict, List, Any


class BraveSearch:
    def __init__(self, api_key: Optional[str] = None):

        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("BRAVE_API_KEY")

    def get_function_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "brave_search",
                "description": "Search the web using Brave Search API for the provided keywords and region",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "string", "description": "The keywords to search for"},
                        "max_results": {
                            "type": "integer",
                            "description": "The maximum number of search results to return",
                            "default": 5,
                        },
                        "region": {
                            "type": "string",
                            "description": "The region to search in. Examples: 'us-en' for United States, 'uk-en' for United Kingdom, 'wt-wt' for No region",
                            "default": "wt-wt",
                        },
                    },
                    "required": ["keywords"],
                },
            },
        }

    def search(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ) -> Union[List[Dict], Dict]:
        """
        Queries the Brave Search API for the provided keywords and region.

        Args:
            keywords (str): The keywords to search for.
            max_results (int, optional): The maximum number of search results to return. Defaults to 10.
            region (str, optional): The region to search in. Defaults to "wt-wt". Possible values include:
                - xa-ar for Arabia
                - xa-en for Arabia (en)
                - ar-es for Argentina
                - au-en for Australia
                - at-de for Austria
                - be-fr for Belgium (fr)
                - be-nl for Belgium (nl)
                - br-pt for Brazil
                - bg-bg for Bulgaria
                - ca-en for Canada
                - ca-fr for Canada (fr)
                - ct-ca for Catalan
                - cl-es for Chile
                - cn-zh for China
                - co-es for Colombia
                - hr-hr for Croatia
                - cz-cs for Czech Republic
                - dk-da for Denmark
                - ee-et for Estonia
                - fi-fi for Finland
                - fr-fr for France
                - de-de for Germany
                - gr-el for Greece
                - hk-tzh for Hong Kong
                - hu-hu for Hungary
                - in-en for India
                - id-id for Indonesia
                - id-en for Indonesia (en)
                - ie-en for Ireland
                - il-he for Israel
                - it-it for Italy
                - jp-jp for Japan
                - kr-kr for Korea
                - lv-lv for Latvia
                - lt-lt for Lithuania
                - xl-es for Latin America
                - my-ms for Malaysia
                - my-en for Malaysia (en)
                - mx-es for Mexico
                - nl-nl for Netherlands
                - nz-en for New Zealand
                - no-no for Norway
                - pe-es for Peru
                - ph-en for Philippines
                - ph-tl for Philippines (tl)
                - pl-pl for Poland
                - pt-pt for Portugal
                - ro-ro for Romania
                - ru-ru for Russia
                - sg-en for Singapore
                - sk-sk for Slovak Republic
                - sl-sl for Slovenia
                - za-en for South Africa
                - es-es for Spain
                - se-sv for Sweden
                - ch-de for Switzerland (de)
                - ch-fr for Switzerland (fr)
                - ch-it for Switzerland (it)
                - tw-tzh for Taiwan
                - th-th for Thailand
                - tr-tr for Turkey
                - ua-uk for Ukraine
                - uk-en for United Kingdom
                - us-en for United States
                - ue-es for United States (es)
                - ve-es for Venezuela
                - vn-vi for Vietnam
                - wt-wt for No region

        Returns:
            list: A list of search result dictionaries, each containing:
                - 'title' (str): The title of the search result.
                - 'href' (str): The URL of the search result.
                - 'body' (str): A brief description or snippet from the search result.
            Or a dict with 'error' key if an error occurred.
        """
        brave_api_key = os.getenv("BRAVE_API_KEY")

        if not brave_api_key:
            return {"error": "No BRAVE_API_KEY environment variable found. Please set it to use this function."}

        backoff = 2  # initial back-off in seconds

        # Map region codes to Brave Search country codes (ISO 3166-1 alpha-2)
        region_mapping = {
            "xa-ar": "SA",
            "xa-en": "SA",
            "ar-es": "AR",
            "au-en": "AU",
            "at-de": "AT",
            "be-fr": "BE",
            "be-nl": "BE",
            "br-pt": "BR",
            "bg-bg": "BG",
            "ca-en": "CA",
            "ca-fr": "CA",
            "ct-ca": "ES",
            "cl-es": "CL",
            "cn-zh": "CN",
            "co-es": "CO",
            "hr-hr": "HR",
            "cz-cs": "CZ",
            "dk-da": "DK",
            "ee-et": "EE",
            "fi-fi": "FI",
            "fr-fr": "FR",
            "de-de": "DE",
            "gr-el": "GR",
            "hk-tzh": "HK",
            "hu-hu": "HU",
            "in-en": "IN",
            "id-id": "ID",
            "id-en": "ID",
            "ie-en": "IE",
            "il-he": "IL",
            "it-it": "IT",
            "jp-jp": "JP",
            "kr-kr": "KR",
            "lv-lv": "LV",
            "lt-lt": "LT",
            "xl-es": "MX",
            "my-ms": "MY",
            "my-en": "MY",
            "mx-es": "MX",
            "nl-nl": "NL",
            "nz-en": "NZ",
            "no-no": "NO",
            "pe-es": "PE",
            "ph-en": "PH",
            "ph-tl": "PH",
            "pl-pl": "PL",
            "pt-pt": "PT",
            "ro-ro": "RO",
            "ru-ru": "RU",
            "sg-en": "SG",
            "sk-sk": "SK",
            "sl-sl": "SI",
            "za-en": "ZA",
            "es-es": "ES",
            "se-sv": "SE",
            "ch-de": "CH",
            "ch-fr": "CH",
            "ch-it": "CH",
            "tw-tzh": "TW",
            "th-th": "TH",
            "tr-tr": "TR",
            "ua-uk": "UA",
            "uk-en": "GB",
            "us-en": "US",
            "ue-es": "US",
            "ve-es": "VE",
            "vn-vi": "VN",
            "wt-wt": "ALL",
        }

        country = region_mapping.get(region, "ALL")

        headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": brave_api_key}

        params = {
            "q": keywords,
            "count": max_results,
            "search_lang": "en",
            "country": country,
        }

        # Infinite retry loop with exponential backoff for rate limits
        while True:
            try:
                response = requests.get(
                    "https://api.search.brave.com/res/v1/web/search", headers=headers, params=params, timeout=30
                )
                response.raise_for_status()
                break  # Success
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    wait_time = backoff + random.uniform(0, backoff)
                    print(f"⚠️ Rate limit hit (429). Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)  # cap the back-off at 2 minutes
                    continue
                else:
                    return {"error": f"HTTP error occurred: {str(e)}"}
            except Exception as e:
                return {"error": f"An error occurred: {str(e)}"}

        try:
            search_results = response.json()
        except Exception as e:
            return {"error": f"Failed to parse response JSON: {str(e)}"}

        if "web" not in search_results or "results" not in search_results["web"]:
            return {"error": "No results found in the response."}

        web_results = search_results["web"]["results"]

        # Convert the search results to the desired format
        results = []
        for result in web_results[:max_results]:
            results.append(
                {
                    "title": result.get("title", ""),
                    "href": result.get("url", ""),
                    "body": result.get("description", ""),
                }
            )

        return results
