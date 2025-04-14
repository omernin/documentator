# src/documentator/main.py

import os
import re
import time
import requests
import logging
import argparse
import json
import sys
from urllib.parse import urljoin, urlparse
from pathlib import Path
from bs4 import BeautifulSoup
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from readability import Document
from tqdm import tqdm

# --- Configuration (Defaults, can be overridden by args) ---
DEFAULT_OUTPUT_DIR = Path("./")
DEFAULT_MAX_PAGES = 50
DEFAULT_REQUEST_DELAY = 1.0
DEFAULT_LLM_MODEL = "gpt-4o-mini"
LOG_FILE = "docdl.log"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"

# --- Logging Setup ---
logger = logging.getLogger("documentator")

# --- OpenAI Client Setup ---
openai_client = None

def initialize_openai_client():
    # ... (no changes needed) ...
    global openai_client
    if openai_client is None:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OpenAI API key not found. Set OPENAI_API_KEY in your .env file.")
            return False
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            logger.debug("OpenAI client initialized.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    return True

# --- Logging Configuration Function ---
def setup_logging(console_log_level=logging.WARNING, log_file=LOG_FILE): # Default console to WARNING
    """Configures file and console logging."""
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler (always DEBUG)
    try:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger ({log_file}): {e}", file=sys.stderr)

    # Console Handler (level based on input, default WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.debug(f"Logging initialized. Console level: {logging.getLevelName(console_log_level)}, File logging to: {log_file}")

# --- Helper Functions ---
# ... (is_valid_url, sanitize_filename, fetch_url, parse_links,
#      get_relevant_links_batch_llm - no changes needed) ...
def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)

def sanitize_filename(url):
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if not path: path = "index"
    filename = re.sub(r'[<>:"/\\|?*]', '_', path)
    filename = filename[:100]
    if filename.lower().endswith(('.md', '.html', '.htm', '.php', '.asp', '.aspx')):
        filename = Path(filename).stem
    if parsed.query:
        query_part = re.sub(r'[^a-zA-Z0-9_-]', '_', parsed.query)[:30]
        filename = f"{filename}_{query_part}"
    filename += ".md"
    return filename

def fetch_url(url):
    headers = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            logger.debug(f"Skipping non-HTML content at {url} (Content-Type: {content_type})")
            return None
        return response.text
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}") # Keep as warning
        return None

def parse_links(html_content, base_url):
    soup = BeautifulSoup(html_content, 'lxml')
    links_data = []
    base_domain = urlparse(base_url).netloc
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.strip().lower().startswith(('mailto:', 'tel:', 'javascript:')): continue
        absolute_url = urljoin(base_url, href)
        parsed_absolute = urlparse(absolute_url)
        if (is_valid_url(absolute_url) and
            parsed_absolute.scheme in ['http', 'https'] and
            parsed_absolute.netloc == base_domain and
            not parsed_absolute.fragment):
              link_text = a_tag.get_text(strip=True) or "N/A"
              links_data.append((absolute_url, link_text))
    return links_data

def get_relevant_links_batch_llm(current_url, candidate_links_data, start_url_domain, start_url_path_prefix, model_name):
    # ... (no changes needed in internal logic) ...
    if not candidate_links_data: return []
    if not initialize_openai_client(): return []
    formatted_links = "\n".join([f"- URL: {url}\n  Text: \"{text}\"" for url, text in candidate_links_data])
    prompt = f"""
    You are an assistant helping to crawl documentation for a specific topic/project.
    The crawling started at a page related to: {start_url_domain}{start_url_path_prefix}
    The current page is: {current_url}
    Candidate Links:
    {formatted_links}
    Respond ONLY with a JSON object containing a single key "relevant_urls" whose value is a list of strings, where each string is a URL from the candidate list that you determined to be relevant.
    Example response format: {{ "relevant_urls": ["url1", "url3", ...] }}
    If no links are relevant, return: {{ "relevant_urls": [] }}
    """
    try:
        logger.debug(f"Sending {len(candidate_links_data)} candidate links to LLM for batch check...")
        completion = openai_client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": "Respond only with the specified JSON format."}, {"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.1)
        response_content = completion.choices[0].message.content
        logger.debug(f"LLM Raw Response: {response_content}")
        response_data = json.loads(response_content)
        relevant_urls = response_data.get("relevant_urls", [])
        if not isinstance(relevant_urls, list):
            logger.warning("LLM returned 'relevant_urls' but not a list. Treating as no relevant links.")
            return []
        valid_relevant_urls = [url for url in relevant_urls if isinstance(url, str) and any(url == candidate_url for candidate_url, _ in candidate_links_data)]
        if len(valid_relevant_urls) != len(relevant_urls):
            logger.warning("LLM response contained URLs not present in the original candidate list. Filtering out.")
        logger.debug(f"LLM identified {len(valid_relevant_urls)} relevant links in batch.")
        return valid_relevant_urls
    except json.JSONDecodeError: logger.error(f"Failed to decode JSON from LLM: {response_content}"); return []
    except RateLimitError: logger.warning("OpenAI rate limit hit. Waiting 60s..."); time.sleep(60); return []
    except APIError as e: logger.error(f"OpenAI API error during batch check: {e}"); return []
    except Exception as e: logger.error(f"Unexpected error during batch LLM check: {e}"); return []

def save_content(url, html_content, domain_output_dir):
    filename = sanitize_filename(url)
    filepath = domain_output_dir / filename
    extracted_text = ""
    try:
        # ... (extraction logic is the same) ...
        doc = Document(html_content)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, 'lxml')
        extracted_text = soup.get_text(separator='\n', strip=True)
        if not extracted_text:
            logger.warning(f"Readability extracted empty content for {url}. Falling back to body text.")
            original_soup = BeautifulSoup(html_content, 'lxml')
            body = original_soup.find('body')
            extracted_text = body.get_text(separator='\n', strip=True) if body else ""
    except Exception as e:
        logger.error(f"Error extracting text for {url}: {e}. Falling back to body text.")
        try:
             original_soup = BeautifulSoup(html_content, 'lxml')
             body = original_soup.find('body')
             extracted_text = body.get_text(separator='\n', strip=True) if body else ""
        except Exception as fallback_e:
             logger.error(f"Error during fallback extraction for {url}: {fallback_e}")
             extracted_text = f"Error extracting content. Original URL: {url}"

    try:
        domain_output_dir.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Original URL: {url}\n\n")
            f.write(extracted_text)
        # --- Use tqdm.write instead of logger.info for console ---
        # Shorten the URL for display if too long
        display_url_save = url if len(url) < 50 else url[:47] + "..."
        tqdm.write(f"Saved: {filepath.name} (from {display_url_save})")
        # --- Log detail to file ---
        logger.debug(f"Saved extracted text from {url} to {filepath}")
        return filepath
    except OSError as e: logger.error(f"Failed to save file {filepath}: {e}"); return None
    except Exception as e: logger.error(f"Unexpected error saving file {filepath}: {e}"); return None

def generate_readme(saved_files, domain_output_dir, start_url):
    readme_path = domain_output_dir / "README.md"
    try:
        with open(readme_path, 'w', encoding='utf-8') as f:
             # ... (readme content is the same) ...
            f.write(f"# Downloaded Documentation Index\n\n")
            f.write(f"Source Domain: {urlparse(start_url).netloc}\n")
            f.write(f"Start URL: {start_url}\n\n")
            f.write("| Local File | Original URL |\n")
            f.write("|------------|--------------|\n")
            for filepath, original_url in sorted(saved_files.items(), key=lambda item: item[0].name):
                relative_path_name = filepath.name
                f.write(f"| [{relative_path_name}]({relative_path_name}) | {original_url} |\n")
        # --- Use tqdm.write instead of logger.info for console ---
        tqdm.write(f"Generated index: {readme_path.relative_to(Path.cwd())}") # Show relative path
         # --- Log detail to file ---
        logger.debug(f"Generated index file at {readme_path}")
    except OSError as e: logger.error(f"Failed to write README.md: {e}")
    except Exception as e: logger.error(f"Unexpected error generating README.md: {e}")


# --- Main Crawling Logic ---
def crawl_and_download(start_url, base_output_dir, max_pages, request_delay, model_name, skip_llm):
    # ... (initial setup and directory creation is the same) ...
    if not is_valid_url(start_url): logger.error(f"Invalid start URL: {start_url}"); return
    start_parsed = urlparse(start_url)
    start_domain = start_parsed.netloc
    if not start_domain: logger.error(f"Could not determine domain: {start_url}"); return
    safe_domain_name = start_domain.replace('.', '_')
    domain_output_dir = base_output_dir / safe_domain_name
    # Use standard logger for this initial setup message
    logger.info(f"Using domain-specific output directory: {domain_output_dir.resolve()}")

    start_path_prefix = '/'.join(start_parsed.path.split('/')[:2]) if start_parsed.path else '/'
    queue = [start_url]
    visited = set()
    saved_files = {}
    pages_processed = 0

    if not skip_llm and not initialize_openai_client():
        logger.error("Exiting: LLM required but OpenAI client failed.")
        return

    with tqdm(total=max_pages, unit="page", desc="Initializing", ascii=" ▖▘▝▗▚▞█", leave=True) as pbar: # leave=True keeps final bar
        while queue and pages_processed < max_pages:
            current_url = queue.pop(0)

            if current_url in visited: logger.debug(f"Skipping visited: {current_url}"); continue
            if urlparse(current_url).netloc != start_domain:
                logger.debug(f"Skipping domain: {current_url}"); visited.add(current_url); continue

            display_url = current_url if len(current_url) < 60 else current_url[:57] + "..."
            pbar.set_description(f"Processing {display_url}")
            visited.add(current_url)

            html_content = fetch_url(current_url)
            if not html_content: time.sleep(request_delay); continue

            # save_content now uses tqdm.write internally for its success message
            filepath = save_content(current_url, html_content, domain_output_dir)
            if filepath:
                saved_files[filepath] = current_url
                pages_processed += 1
                pbar.update(1)
            # else: # If save failed, we logged the error in save_content

            # --- Link processing ---
            all_links_data = parse_links(html_content, current_url)
            logger.debug(f"Found {len(all_links_data)} potential links on {current_url}")

            candidate_links_data = []
            candidate_urls_set = set()
            for link_url, link_text in all_links_data:
                if link_url not in visited and link_url not in queue and link_url not in candidate_urls_set:
                    if urlparse(link_url).netloc == start_domain:
                        candidate_links_data.append((link_url, link_text))
                        candidate_urls_set.add(link_url)

            if not candidate_links_data:
                logger.debug("No new candidate links found.")
                time.sleep(request_delay)
                continue

            relevant_urls_to_add = []
            if skip_llm:
                logger.debug("LLM skipped. Adding all new internal links.")
                relevant_urls_to_add = [url for url, text in candidate_links_data]
            else:
                logger.debug(f"Checking {len(candidate_links_data)} candidates via LLM...")
                relevant_urls_to_add = get_relevant_links_batch_llm(
                    current_url, candidate_links_data, start_domain, start_path_prefix, model_name
                )

            added_count = 0
            for relevant_url in relevant_urls_to_add:
                if relevant_url not in queue:
                    # Limit queue size reasonably
                    if len(queue) < (max_pages * 10): # Increased buffer
                         queue.append(relevant_url)
                         added_count += 1
                    else:
                        logger.warning("Queue size limit reached, stopping adding links for this page.")
                        break # Stop adding if queue seems excessively large
            logger.debug(f"Added {added_count} links to queue.")
            # --- End Link Processing ---

            time.sleep(request_delay) # Delay after processing each page

        # --- End of While Loop ---
        pbar.set_description(f"Finished ({pages_processed}/{max_pages})")
        pbar.n = pages_processed # Ensure final count is accurate
        pbar.refresh()


    # Use standard logger for final messages after progress bar is done
    logger.info(f"Crawling finished. Successfully processed and saved {pages_processed} pages for domain {start_domain}.")
    if saved_files:
        # generate_readme now uses tqdm.write internally
        generate_readme(saved_files, domain_output_dir, start_url)
    else:
        logger.info(f"No files were downloaded into {domain_output_dir}.")


# --- Argument Parsing and Script Execution ---

def parse_arguments():
    # ... (no changes needed) ...
    parser = argparse.ArgumentParser(description="Crawl and download documentation from a starting URL.")
    parser.add_argument("start_url", metavar="START_URL", type=str, help="The initial URL.")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Base directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("-m", "--max-pages", type=int, default=DEFAULT_MAX_PAGES, help=f"Max pages (default: {DEFAULT_MAX_PAGES})")
    parser.add_argument("--delay", type=float, default=DEFAULT_REQUEST_DELAY, help=f"Delay between requests (default: {DEFAULT_REQUEST_DELAY})")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL, help=f"OpenAI model (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM relevance checks.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging to console and file.")
    return parser.parse_args()


def cli_entry_point():
    args = parse_arguments()

    # Setup Logging FIRST - Default console level is WARNING now
    console_level = logging.DEBUG if args.verbose else logging.WARNING
    setup_logging(console_log_level=console_level, log_file=LOG_FILE)

    # ... (validation and initial logging remain the same) ...
    if args.max_pages <= 0: exit(logger.error("--max-pages must be positive."))
    if args.delay < 0: exit(logger.error("--delay cannot be negative."))
    if not args.start_url or not is_valid_url(args.start_url):
         exit(logger.error("A valid start URL is required."))
    try: args.output.mkdir(parents=True, exist_ok=True)
    except OSError as e: exit(logger.error(f"Failed to create base output directory {args.output}: {e}"))

    logger.info("-" * 20 + " Starting Download " + "-" * 20)
    logger.info(f"Start URL: {args.start_url}")
    logger.info(f"Base Output Directory: {args.output.resolve()}")
    logger.info(f"Max Pages: {args.max_pages}")
    logger.info(f"Request Delay: {args.delay}s")
    if args.no_llm: logger.info("LLM Relevance Check: DISABLED")
    else: logger.info(f"LLM Model: {args.model}")
    logger.info(f"Log File: {Path(LOG_FILE).resolve()}")


    crawl_and_download(
        start_url=args.start_url,
        base_output_dir=args.output,
        max_pages=args.max_pages,
        request_delay=args.delay,
        model_name=args.model,
        skip_llm=args.no_llm
    )
    logger.info("-" * 20 + " Download Complete " + "-" * 20)


if __name__ == "__main__":
    cli_entry_point()