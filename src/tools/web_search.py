"""
Web search and data retrieval utilities for agent research capabilities.
Implements search functionality for academic papers, general web search, and more.
Inspired by the AgentLaboratory implementation.
"""
import requests
import re
import time
import random
from typing import List, Dict, Any, Optional
import arxiv
from urllib.parse import quote_plus
import feedparser
from bs4 import BeautifulSoup


# Default headers for web requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0"
}

class ArxivSearch:
    """Class for searching and retrieving papers from arXiv."""
    
    def __init__(self, max_results: int = 5):
        """
        Initialize the ArXiv searcher.
        
        Args:
            max_results: Maximum number of results to return per search
        """
        self.max_results = max_results
        
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (overrides instance default)
            
        Returns:
            List of paper dictionaries with metadata
        """
        if max_results is None:
            max_results = self.max_results
            
        try:
            # Use the arxiv API to search for papers
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                # Extract and format the paper information
                authors = [author.name for author in paper.authors]
                result = {
                    "title": paper.title,
                    "summary": paper.summary,
                    "authors": authors,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "arxiv_id": paper.get_short_id(),
                    "pdf_url": paper.pdf_url,
                    "url": paper.entry_id,
                    "source": "arxiv"
                }
                results.append(result)
                
            return results
        
        except Exception as e:
            print(f"ArXiv search error: {str(e)}")
            # Fallback to RSS feed method if the API fails
            return self._search_via_rss(query, max_results)
    
    def _search_via_rss(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Fallback method to search arXiv via its RSS feed.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of paper dictionaries with metadata
        """
        try:
            # Format the query for the arXiv API
            query = query.replace(' ', '+')
            url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
            
            # Get the RSS feed
            response = requests.get(url, headers=DEFAULT_HEADERS)
            response.raise_for_status()
            
            # Parse the feed
            feed = feedparser.parse(response.content)
            
            results = []
            for entry in feed.entries:
                # Extract authors
                authors = [author.name for author in entry.get('authors', [])]
                
                # Get PDF link
                pdf_link = next((link.href for link in entry.links if link.get('title') == 'pdf'), None)
                
                # Create the result dictionary
                result = {
                    'title': entry.get('title', 'Untitled').replace('\n', ' '),
                    'authors': authors,
                    'summary': entry.get('summary', '').replace('\n', ' '),
                    'published': entry.get('published', ''),
                    'arxiv_id': entry.get('id', '').split('/')[-1],
                    'pdf_url': pdf_link,
                    'url': entry.get('link', ''),
                    'source': 'arxiv'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"ArXiv RSS search error: {str(e)}")
            return []
    
    def get_paper_text(self, paper_id: str) -> str:
        """
        Get the full text of a paper by its arXiv ID.
        
        Args:
            paper_id: arXiv ID of the paper
            
        Returns:
            Paper text or empty string if retrieval fails
        """
        try:
            from ..utils.document_loader import fetch_paper_content
            
            # Use the centralized paper retrieval function
            paper_text = fetch_paper_content(paper_id, source="arxiv")
            
            return paper_text
        
        except Exception as e:
            print(f"Error getting paper text: {str(e)}")
            return ""

class BioRxivSearch:
    """Class for searching and retrieving papers from bioRxiv."""
    
    def __init__(self, max_results: int = 5):
        """
        Initialize the bioRxiv searcher.
        
        Args:
            max_results: Maximum number of results to return per search
        """
        self.max_results = max_results
        self.base_url = "https://api.biorxiv.org/details/biorxiv"
    
    def search(self, query: str, max_results: Optional[int] = None, 
             max_retries: int = 5, initial_delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Search bioRxiv for papers matching the query with exponential backoff retries.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (overrides instance default)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before retrying
            
        Returns:
            List of paper dictionaries with metadata
        """
        if max_results is None:
            max_results = self.max_results
        
        # Implement retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                # Attempt the scraping search
                results = self._search_via_scraping(query, max_results)
                
                # If we got results, return them
                if results and len(results) > 0:
                    if attempt > 0:
                        print(f"✅ Successfully found {len(results)} results from bioRxiv on attempt {attempt+1}")
                    return results
                
                # If no results but no error, try with a modified query
                if attempt < max_retries - 1:
                    # Calculate delay with exponential backoff
                    delay = initial_delay * (2 ** attempt)
                    print(f"⚠️ No bioRxiv results found for '{query}'. Retrying in {delay:.1f}s... (Attempt {attempt+1}/{max_retries})")
                    
                    # Try different query strategies on subsequent attempts
                    if " AND " in query or " OR " in query:
                        # Simplify complex queries
                        query = query.replace(" AND ", " ").replace(" OR ", " ")
                    elif len(query.split()) > 2:
                        # Remove the least important word for next attempt
                        words = query.split()
                        stopwords = ["the", "a", "an", "in", "on", "of", "and", "for", "with", "to"]
                        # First try removing stopwords if present
                        for stopword in stopwords:
                            if stopword in words:
                                words.remove(stopword)
                                break
                        else:
                            # Otherwise remove the last word
                            words = words[:-1]
                        query = " ".join(words)
                    
                    time.sleep(delay)
            
            except requests.exceptions.RequestException as e:
                # Handle network-related errors specifically
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"⚠️ bioRxiv request failed: {str(e)}. Retrying in {delay:.1f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"❌ bioRxiv search failed after {max_retries} attempts: {str(e)}")
                    return []
                    
            except Exception as e:
                # Handle other errors
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"⚠️ bioRxiv search error: {str(e)}. Retrying in {delay:.1f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"❌ bioRxiv search failed after {max_retries} attempts: {str(e)}")
                    return []
        
        # If we've exhausted all retries
        print(f"⚠️ No results found from bioRxiv after {max_retries} attempts with query: '{query}'")
        return []
    
    def _search_via_scraping(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Search bioRxiv for papers via web scraping with improved robustness.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of paper dictionaries with metadata
        """
        search_query = quote_plus(query)
        url = f"https://www.biorxiv.org/search/{search_query}"
        
        # Create a session to handle cookies
        session = requests.Session()
        # Use a more browser-like user agent with version randomization to avoid detection
        user_agent = f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90, 120)}.0.0.0 Safari/537.36"
        session.headers.update({
            **DEFAULT_HEADERS,
            "User-Agent": user_agent
        })
        
        # First visit homepage to get cookies and establish a normal browsing pattern
        try:
            homepage_response = session.get("https://www.biorxiv.org/", timeout=15)
            homepage_response.raise_for_status()
            # Add a slightly variable delay to mimic human behavior
            time.sleep(random.uniform(1.5, 3))
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not access bioRxiv homepage: {str(e)}")
            # Continue anyway - we might still succeed with direct search
        
        # Get the search results page using the session
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        # Check if we got a captcha or empty page
        if "captcha" in response.text.lower() or len(response.content) < 1000:
            raise Exception("Possible rate limiting or captcha detected. Response too short or contains captcha")
            
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if we have search results section
        search_container = soup.select_one(".search-results") or soup.select_one("#search-results")
        if not search_container:
            # Try alternative detection - look for common page elements
            has_content = bool(soup.select_one("form#search-block-form"))
            if not has_content:
                raise Exception("Search results page structure not recognized")
        
        # Find all search result items (try multiple selectors)
        result_items = soup.select(".search-result")
        if not result_items:
            result_items = soup.select(".highwire-article-wrapper")
        if not result_items:
            result_items = soup.select("li.search-result")
        if not result_items:
            # Try even more general selectors as last resort
            result_items = soup.select(".item-list li") or soup.select("article")
            
        # Log details for debugging
        if not result_items:
            # Save response content to analyze structure
            debug_info = f"bioRxiv search response length: {len(response.text)}"
            # Check page title for clues
            title_elem = soup.select_one("title")
            if title_elem:
                debug_info += f", Page title: {title_elem.get_text()}"
            print(f"Debug: {debug_info}")
            
        results = []
        for item in result_items[:max_results]:
            try:
                # Extract title
                title_elem = (item.select_one(".highwire-cite-title") or 
                              item.select_one("h2") or 
                              item.select_one(".title") or
                              item.select_one("h3") or
                              item.select_one("a[href*='/content/']"))
                
                title = title_elem.get_text(strip=True) if title_elem else "Untitled"
                
                # Extract URL
                url = None
                url_elem = title_elem.select_one("a") if title_elem else None
                if url_elem and url_elem.has_attr('href'):
                    url_path = url_elem['href']
                    # Make sure it's an absolute URL
                    if url_path.startswith('http'):
                        url = url_path
                    else:
                        url = f"https://www.biorxiv.org{url_path}"
                elif title_elem and title_elem.name == "a" and title_elem.has_attr('href'):
                    url_path = title_elem['href']
                    if url_path.startswith('http'):
                        url = url_path
                    else:
                        url = f"https://www.biorxiv.org{url_path}"
                
                # If still no URL, search for any link
                if not url:
                    any_link = item.select_one("a[href*='biorxiv']")
                    if any_link and any_link.has_attr('href'):
                        url_path = any_link['href']
                        if url_path.startswith('http'):
                            url = url_path
                        else:
                            url = f"https://www.biorxiv.org{url_path}"
                
                # Extract authors
                authors_elem = (item.select_one(".highwire-citation-authors") or 
                                item.select_one(".meta-authors") or
                                item.select_one(".authors"))
                authors = []
                if authors_elem:
                    # Try multiple author selectors
                    author_links = (authors_elem.select("span.highwire-citation-author") or 
                                    authors_elem.select("span.author") or
                                    authors_elem.select("a[href*='search-author']"))
                    if author_links:
                        authors = [author.get_text(strip=True) for author in author_links]
                    else:
                        # If no spans found, try getting the text directly
                        authors_text = authors_elem.get_text(strip=True)
                        if authors_text:
                            # Split by commas for multiple authors
                            authors = [a.strip() for a in authors_text.split(',')]
                
                # Extract abstract/summary
                abstract_elem = (item.select_one(".highwire-cite-snippet") or 
                                 item.select_one(".meta-abstract") or 
                                 item.select_one(".abstract") or
                                 item.select_one("p"))
                abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
                
                # Extract publication date
                date_elem = (item.select_one(".highwire-cite-metadata-date") or 
                             item.select_one(".meta-pub-date") or
                             item.select_one(".date"))
                published = date_elem.get_text(strip=True) if date_elem else ""
                
                # If no date found, try to extract from URL or DOI
                if not published:
                    # Look for date in URL format (common pattern in bioRxiv URLs)
                    if url:
                        # bioRxiv URLs typically contain the DOI with date: 10.1101/YYYY.MM.DD.NNNNNN
                        url_date_match = re.search(r'/10\.1101/(\d{4}\.\d{2}\.\d{2})\.', url)
                        if url_date_match:
                            date_str = url_date_match.group(1)
                            # Convert to more readable format (YYYY-MM-DD)
                            try:
                                parts = date_str.split('.')
                                if len(parts) == 3:
                                    published = f"{parts[0]}-{parts[1]}-{parts[2]}"
                            except:
                                published = date_str
                
                # Extract DOI
                doi = ""
                doi_elem = (item.select_one(".highwire-cite-metadata-doi") or 
                            item.select_one(".meta-doi") or
                            item.select_one("[data-doi]"))
                if doi_elem:
                    if doi_elem.has_attr('data-doi'):
                        doi = doi_elem['data-doi']
                    else:
                        doi_text = doi_elem.get_text(strip=True)
                        doi_match = re.search(r'doi:\s*([\d\.]+/[\w\.]+)', doi_text)
                        if doi_match:
                            doi = doi_match.group(1)
                
                # If no DOI yet, try to extract from URL
                if not doi and url:
                    url_match = re.search(r'biorxiv\.org/content/([\d\.]+/[\w\.]+)', url)
                    if url_match:
                        doi = url_match.group(1)
                
                # Ensure we have a valid URL
                if not url and doi:
                    url = f"https://www.biorxiv.org/content/{doi}"
                
                # Skip if we don't have enough info to identify the paper
                if not title or title == "Untitled" and not url and not doi:
                    continue
                
                # Create result dictionary
                result = {
                    "title": title,
                    "summary": abstract,
                    "authors": authors,
                    "published": published,
                    "doi": doi,
                    "url": url,
                    "pdf_url": f"{url}.full.pdf" if url else "",
                    "source": "biorxiv"
                }
                results.append(result)
                
            except Exception as e:
                # If an error occurs processing one result, continue with others
                print(f"Warning: Error extracting paper details: {str(e)}")
                continue
        
        if not results:
            print(f"bioRxiv search for '{query}' returned no results using web scraping")
        else:
            print(f"✅ Successfully extracted {len(results)} results from bioRxiv")
            
        return results
    
    def get_paper_text(self, paper_id: str, max_retries: int = 3) -> str:
        """
        Get the full text of a paper by its bioRxiv DOI or URL with retry logic.
        
        Args:
            paper_id: DOI or URL of the paper
            max_retries: Maximum number of retry attempts
            
        Returns:
            Paper text or error message if retrieval fails after retries
        """
        from ..utils.document_loader import fetch_paper_content
        
        for attempt in range(max_retries):
            try:
                # Use the centralized paper retrieval function
                paper_text = fetch_paper_content(paper_id, source="biorxiv")
                
                # Check if we got meaningful content or an error message
                if paper_text and not paper_text.startswith("Error"):
                    if attempt > 0:
                        print(f"✅ Successfully retrieved bioRxiv paper content on attempt {attempt+1}")
                    return paper_text
                
                # If we got an error but have retries left
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"⚠️ Failed to retrieve bioRxiv paper (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
                    time.sleep(delay)
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"⚠️ Error retrieving bioRxiv paper: {str(e)}. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"❌ Failed to retrieve bioRxiv paper after {max_retries} attempts: {str(e)}")
                    return f"Error retrieving bioRxiv paper after multiple attempts: {str(e)}"
        
        return f"Error retrieving bioRxiv paper content after {max_retries} attempts"


def fetch_webpage_content(url: str, extract_text: bool = True) -> Dict[str, Any]:
    """
    Fetch the content of a webpage, with support for PDF files.
    
    Args:
        url: URL to fetch
        extract_text: Whether to extract main text content from HTML
        
    Returns:
        Dictionary with webpage content and metadata
    """
    try:
        from ..utils.document_loader import _fetch_url_content
        
        # Use the centralized URL content fetching function
        content = _fetch_url_content(url)
        
        # Parse the content and extract the relevant parts
        lines = content.split("\n")
        title = ""
        body = []
        
        # Extract title from the first line
        if lines and lines[0].startswith("Title:"):
            title = lines[0].replace("Title:", "").strip()
            lines = lines[2:]  # Skip the title and the blank line after it
        
        # The rest is the content
        body = "\n".join(lines)
        
        result = {
            "url": url,
            "status_code": 200,  # We assume success since _fetch_url_content returns error messages on failure
            "content_type": "text/html" if not url.lower().endswith('.pdf') else "application/pdf",
            "raw_html": None,
            "content": body,
            "title": title,
            "error": None
        }
        
        # Check if the content indicates an error
        if "Error fetching URL content:" in content or "Error processing" in content:
            result["error"] = content
            result["status_code"] = None
        
        return result
    
    except Exception as e:
        return {
            "url": url,
            "status_code": None,
            "content_type": None,
            "raw_html": None,
            "content": None,
            "title": None,
            "error": str(e)
        }