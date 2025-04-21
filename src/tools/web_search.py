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
    """Class for searching and retrieving papers from bioRxiv using the official API."""
    
    def __init__(self, max_results: int = 5):
        """
        Initialize the bioRxiv searcher.
        
        Args:
            max_results: Maximum number of results to return per search
        """
        self.max_results = max_results
        self.base_url = "https://api.biorxiv.org"
    
    def search(self, query: str, max_results: Optional[int] = None, 
               months: int = 12, max_retries: int = 3, 
               initial_delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Search bioRxiv for papers matching the query with API integration.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (overrides instance default)
            months: Number of months to search back from current date (default: 12)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before retrying
            
        Returns:
            List of paper dictionaries with metadata
        """
        if max_results is None:
            max_results = self.max_results
        
        # Search terms for matching
        search_terms = query.strip().lower().split()
        
        # Fetch papers only ONCE, then reuse for all retries
        all_papers = None
        
        # Track matching mode for each attempt
        matching_mode = "all"  # Start with strict matching
        
        # Implement retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                # On first attempt, fetch papers from API
                if attempt == 0:
                    # Calculate date range based on months parameter
                    from datetime import datetime, timedelta
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30 * months)
                    date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
                    
                    # Fetch papers ONLY on first attempt (we'll reuse them for retries)
                    all_papers = self._fetch_papers_from_api(date_range, months)
                    
                    if not all_papers:
                        print(f"Error fetching papers from bioRxiv.")
                        return []
                
                # Filter papers based on current matching mode
                filtered_results = []
                
                # Match papers against search terms
                for paper in all_papers:
                    title = paper.get("title", "").lower()
                    abstract = paper.get("abstract", "").lower()
                    
                    # Different matching strategies based on attempt
                    matches = False
                    if matching_mode == "all":
                        # Require ALL terms to match (strictest)
                        matches = all(term in title or term in abstract for term in search_terms)
                    elif matching_mode == "any":
                        # Require ANY term to match (more lenient)
                        matches = any(term in title or term in abstract for term in search_terms)
                    else:
                        # Most lenient: check substring matches
                        matches = any(term in title or term in abstract for term in search_terms)
                    
                    # Add matching papers to results
                    if matches:
                        doi = paper.get("doi", "")
                        version = paper.get("version", "1")
                        
                        result = {
                            "title": paper.get("title", "Untitled"),
                            "summary": paper.get("abstract", ""),
                            "authors": paper.get("authors", "").split("; "),
                            "published": paper.get("date", ""),
                            "doi": doi,
                            "url": f"https://www.biorxiv.org/content/{doi}v{version}",
                            "pdf_url": f"https://www.biorxiv.org/content/{doi}v{version}.full.pdf",
                            "source": "biorxiv",
                            "version": version,
                            "category": paper.get("category", "")
                        }
                        filtered_results.append(result)
                
                # Remove duplicate papers by DOI, keeping only the latest version
                deduplicated_results = {}
                for paper in filtered_results:
                    doi_base = paper.get("doi", "").split("v")[0]  # Get base DOI without version
                    version = int(paper.get("version", "1"))
                    
                    # If this is a new paper or a newer version, keep it
                    if doi_base not in deduplicated_results or version > int(deduplicated_results[doi_base].get("version", "1")):
                        deduplicated_results[doi_base] = paper
                
                # Convert back to list
                filtered_results = list(deduplicated_results.values())
                
                # Sort by relevance
                def relevance_score(paper):
                    score = 0
                    title = paper["title"].lower()
                    abstract = paper["summary"].lower()
                    
                    for term in search_terms:
                        # Weight title matches more heavily
                        if term in title:
                            score += 10
                        if term in abstract:
                            score += 5
                        
                        # Additional score for frequency
                        score += title.count(term) * 2
                        score += abstract.count(term)
                    
                    return score
                
                filtered_results.sort(key=relevance_score, reverse=True)
                
                # If we found any results, return them
                if filtered_results:
                    if attempt > 0:
                        print(f"✅ Successfully found {len(filtered_results)} results from bioRxiv on retry attempt {attempt+1}")
                    return filtered_results[:max_results]
                
                # No results but still have retries left
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    
                    # Change matching strategy for next attempt
                    if attempt == 0:
                        print(f"⚠️ No bioRxiv results found for '{query}'. Retrying with less strict matching in {delay:.1f}s... (Attempt {attempt+1}/{max_retries})")
                        matching_mode = "any"  # Next attempt: match ANY terms
                    else:
                        print(f"⚠️ Still no results. Trying broader matching in {delay:.1f}s... (Attempt {attempt+1}/{max_retries})")
                        matching_mode = "substring"  # Last attempt: try substring matching
                    
                    time.sleep(delay)
            
            except Exception as e:
                # Handle any errors
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
    
    def _fetch_papers_from_api(self, date_range: str, months: int) -> List[Dict[str, Any]]:
        """
        Fetch papers from bioRxiv API for the given date range.
        
        Args:
            date_range: Range of dates in format "YYYY-MM-DD/YYYY-MM-DD"
            months: Number of months (for logging purposes)
            
        Returns:
            List of paper dictionaries from the API
        """
        # Initialize papers collection
        all_papers = []
        
        # Create a session with appropriate headers
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)
        
        # Initial request - format: /details/biorxiv/[date_range]/[cursor]/json
        cursor = 0
        url = f"{self.base_url}/details/biorxiv/{date_range}/{cursor}/json"
        
        try:
            print(f"Fetching bioRxiv papers from the past {months} months...")
            response = session.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Process results
            if data and "collection" in data and data["collection"]:
                all_papers.extend(data["collection"])
                
                # Get total count from messages
                total_count = 0
                if "messages" in data and data["messages"]:
                    total_count = int(data["messages"][0].get("total", "0"))
                
                print(f"Found {total_count} papers in bioRxiv from the past {months} months")
                
                # Fetch additional pages if needed (bioRxiv API returns ~100 results per page)
                max_papers_to_fetch = min(1000, total_count)  # Limit to reasonable number to search through
                
                while len(all_papers) < max_papers_to_fetch and cursor < total_count:
                    cursor += 100
                    next_page_url = f"{self.base_url}/details/biorxiv/{date_range}/{cursor}/json"
                    
                    try:
                        page_response = session.get(next_page_url, timeout=15)
                        page_response.raise_for_status()
                        
                        page_data = page_response.json()
                        if page_data and "collection" in page_data and page_data["collection"]:
                            all_papers.extend(page_data["collection"])
                    except Exception as e:
                        print(f"Warning: Error fetching additional page: {str(e)}")
                        break
            
        except Exception as e:
            print(f"Error fetching papers from bioRxiv: {str(e)}")
            return []
        
        return all_papers
        
    # We can remove this method since the filtering is now done directly in the search method
    
    def get_paper_text(self, paper_id: str, max_retries: int = 3) -> str:
        """
        Get the full text of a paper by its bioRxiv DOI or URL with retry logic.
        Uses a combination of the API for metadata and document_loader for content.
        
        Args:
            paper_id: DOI or URL of the paper
            max_retries: Maximum number of retry attempts
            
        Returns:
            Paper text or error message if retrieval fails after retries
        """
        from ..utils.document_loader import fetch_paper_content
        
        # Clean up the paper ID
        paper_id = paper_id.strip()
        doi = None
        
        # Extract DOI from different possible formats
        if paper_id.startswith("http"):
            # Extract DOI from URL
            doi_match = re.search(r'biorxiv\.org/content/([\d\.]+/[\w\.]+)', paper_id)
            if doi_match:
                doi = doi_match.group(1)
        elif paper_id.startswith("10."):
            # Direct DOI format
            doi = paper_id
        
        # If we have a DOI, try to get metadata from the API first
        metadata = None
        if doi:
            try:
                # Try to get metadata from the API using the correct endpoint format
                # Format: https://api.biorxiv.org/details/biorxiv/[DOI]/json
                meta_url = f"{self.base_url}/details/biorxiv/{doi}/json"
                
                session = requests.Session()
                session.headers.update(DEFAULT_HEADERS)
                meta_response = session.get(meta_url, timeout=10)
                meta_response.raise_for_status()
                
                meta_data = meta_response.json()
                
                if meta_data.get("collection") and len(meta_data["collection"]) > 0:
                    metadata = meta_data["collection"][0]
                    print(f"✅ Successfully retrieved metadata for bioRxiv paper with DOI: {doi}")
            except Exception as e:
                print(f"Warning: Could not retrieve bioRxiv metadata from API: {str(e)}")
        
        # Now proceed with the document retrieval with retry logic
        for attempt in range(max_retries):
            try:
                # Use the centralized paper retrieval function with optional metadata
                paper_text = fetch_paper_content(paper_id, source="biorxiv", metadata=metadata)
                
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