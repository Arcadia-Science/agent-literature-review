"""
Document loader utility to handle various document formats and online paper retrieval.
"""
import os
import tempfile
import requests
import re
from typing import Optional, Dict, Any

# Default headers for web requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

def load_document(file_path: str) -> Optional[str]:
    """
    Load document content from various file formats.
    
    Args:
        file_path: Path to the document file.
        
    Returns:
        Document content as text if successful, None otherwise.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            return _load_pdf_file(file_path)
        else:
            print(f"Error: Unsupported file format: {file_ext}")
            return None
    except Exception as e:
        print(f"Error loading document {file_path}: {str(e)}")
        return None

def _load_pdf_file(file_path: str) -> str:
    """
    Load content from a PDF file.
    
    Note: This requires PyPDF2. If not installed, it will return an error message.
    """
    try:
        import PyPDF2
    except ImportError:
        return ("Error: PyPDF2 is required to read PDF files. "
                "Install it with 'pip install PyPDF2'.")
    
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    
    return text

def fetch_paper_content(paper_id: str, source: str = "arxiv", metadata: Dict[str, Any] = None) -> str:
    """
    Centralized function to fetch paper content from various sources (arXiv, bioRxiv, URL, or local file).
    
    Args:
        paper_id: Paper ID (arXiv ID, DOI, URL, or local file path)
        source: Source of the paper ('arxiv', 'biorxiv', 'url', or 'local_folder')
        metadata: Optional metadata about the paper (title, authors, etc.)
        
    Returns:
        Paper content as text, or error message if retrieval fails
    """
    try:
        # Clean up the paper ID
        paper_id = paper_id.strip()
        
        # Handle arXiv papers
        if source.lower() == "arxiv":
            return _fetch_arxiv_paper(paper_id, metadata)
        
        # Handle bioRxiv papers
        elif source.lower() == "biorxiv":
            return _fetch_biorxiv_paper(paper_id, metadata)
        
        # Handle local file
        elif source.lower() == "local_folder" and os.path.exists(paper_id):
            return _fetch_local_paper(paper_id, metadata)
        
        # Handle general URL (could be PDF or webpage)
        elif source.lower() == "url" or paper_id.startswith("http"):
            return _fetch_url_content(paper_id, metadata)
        
        else:
            return f"Unsupported paper source: {source}"
    
    except Exception as e:
        return f"Error fetching paper content: {str(e)}"

def _fetch_arxiv_paper(paper_id: str, metadata: Dict[str, Any] = None) -> str:
    """
    Fetch a paper from arXiv by its ID.
    
    Args:
        paper_id: arXiv ID of the paper
        metadata: Optional metadata about the paper
        
    Returns:
        Paper text or error message
    """
    try:
        # Clean up the ID and handle potential URL formats
        if paper_id.startswith("http"):
            # Extract ID from URL
            paper_id = paper_id.split("/")[-1]
        
        # Remove version number if present (e.g., v1, v2)
        if "v" in paper_id and paper_id[-2] == "v" and paper_id[-1].isdigit():
            paper_id = paper_id[:-2]
        
        # Importing arxiv here to avoid dependency issues if not installed
        try:
            import arxiv
        except ImportError:
            return "Error: arxiv package is required to fetch arXiv papers. Install with 'pip install arxiv'."
        
        # Get the paper
        paper = next(arxiv.Search(id_list=[paper_id]).results())
        
        # Try to get the PDF content
        try:
            # Create a temporary directory to store the PDF
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the PDF
                pdf_url = paper.pdf_url
                pdf_path = os.path.join(temp_dir, f"{paper_id}.pdf")
                
                response = requests.get(pdf_url, headers=DEFAULT_HEADERS, stream=True)
                response.raise_for_status()
                
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Parse the PDF
                pdf_text = load_document(pdf_path)
                
                if pdf_text:
                    # Combine metadata with full text
                    return f"Title: {paper.title}\n\nAuthors: {', '.join([a.name for a in paper.authors])}\n\nAbstract: {paper.summary}\n\nFull Text:\n{pdf_text}"
        
        except Exception as pdf_error:
            print(f"Error downloading/parsing PDF: {str(pdf_error)}")
        
        # Return the summary as a fallback if PDF parsing failed
        return f"Title: {paper.title}\n\nAuthors: {', '.join([a.name for a in paper.authors])}\n\nAbstract: {paper.summary}"
    
    except Exception as e:
        return f"Error fetching arXiv paper: {str(e)}"

def _fetch_biorxiv_paper(paper_id: str, metadata: Dict[str, Any] = None) -> str:
    """
    Fetch a paper from bioRxiv by its DOI or URL.
    
    Args:
        paper_id: DOI or URL of the paper
        metadata: Optional metadata about the paper
        
    Returns:
        Paper text or error message
    """
    try:
        # Clean up the paper ID
        paper_id = paper_id.strip()
        doi = None
        
        # Handle different input formats
        if paper_id.startswith("http"):
            # Extract DOI from URL
            doi_match = re.search(r'biorxiv\.org/content/([\d\.]+/[\w\.]+)', paper_id)
            if doi_match:
                doi = doi_match.group(1)
        elif paper_id.startswith("10."):
            # Direct DOI format
            doi = paper_id
        
        if not doi:
            return f"Could not extract a valid DOI from: {paper_id}"
        
        # Get PDF URL
        pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf"
        
        # Try to download and parse the PDF
        try:
            # Create a temporary directory to store the PDF
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the PDF
                pdf_path = os.path.join(temp_dir, f"{doi.replace('/', '_')}.pdf")
                response = requests.get(pdf_url, headers=DEFAULT_HEADERS, stream=True)
                response.raise_for_status()
                
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Parse the PDF
                pdf_text = load_document(pdf_path)
                
                if pdf_text:
                    # Try to get metadata from the API
                    meta_url = f"https://api.biorxiv.org/details/biorxiv/{doi}/json"
                    meta_response = requests.get(meta_url, headers=DEFAULT_HEADERS)
                    meta_data = meta_response.json()
                    
                    title = "Unknown Title"
                    authors = "Unknown Authors"
                    abstract = "No abstract available"
                    
                    if meta_data.get("collection") and len(meta_data["collection"]) > 0:
                        paper_meta = meta_data["collection"][0]
                        title = paper_meta.get("title", "Unknown Title")
                        authors = paper_meta.get("authors", "Unknown Authors")
                        abstract = paper_meta.get("abstract", "No abstract available")
                    
                    # Include metadata and full text
                    return f"Title: {title}\n\nAuthors: {authors}\n\nAbstract: {abstract}\n\nFull Text:\n{pdf_text}"
                else:
                    # If PDF parsing failed, get the HTML version as fallback
                    from bs4 import BeautifulSoup
                    html_url = f"https://www.biorxiv.org/content/{doi}.full"
                    html_response = requests.get(html_url, headers=DEFAULT_HEADERS)
                    html_response.raise_for_status()
                    
                    soup = BeautifulSoup(html_response.content, 'html.parser')
                    
                    # Extract title
                    title_elem = soup.select_one(".article-title")
                    title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"
                    
                    # Extract authors
                    authors_elems = soup.select(".contrib-author")
                    authors = [author.get_text(strip=True) for author in authors_elems]
                    
                    # Extract abstract
                    abstract_elem = soup.select_one(".abstract")
                    abstract = abstract_elem.get_text(strip=True) if abstract_elem else "No abstract available"
                    
                    # Extract main text
                    article_elems = soup.select(".article-section")
                    article_text = "\n\n".join([section.get_text(strip=True) for section in article_elems])
                    
                    # Return formatted text
                    return f"Title: {title}\n\nAuthors: {', '.join(authors)}\n\nAbstract: {abstract}\n\nFull Text:\n{article_text}"
        
        except Exception as pdf_error:
            print(f"Error downloading/parsing PDF: {str(pdf_error)}")
        
        # Fallback to just getting metadata if full text parsing failed
        meta_url = f"https://api.biorxiv.org/details/biorxiv/{doi}/json"
        meta_response = requests.get(meta_url, headers=DEFAULT_HEADERS)
        meta_data = meta_response.json()
        
        if meta_data.get("collection") and len(meta_data["collection"]) > 0:
            paper_meta = meta_data["collection"][0]
            title = paper_meta.get("title", "Unknown Title")
            authors = paper_meta.get("authors", "Unknown Authors")
            abstract = paper_meta.get("abstract", "No abstract available")
            return f"Title: {title}\n\nAuthors: {authors}\n\nAbstract: {abstract}"
        
        return f"Could not retrieve content for bioRxiv paper with DOI: {doi}"
    
    except Exception as e:
        return f"Error fetching bioRxiv paper: {str(e)}"

def _fetch_local_paper(file_path: str, metadata: Dict[str, Any] = None) -> str:
    """
    Load content from a local paper file.
    
    Args:
        file_path: Path to the local file
        metadata: Optional metadata about the paper
        
    Returns:
        Paper content as text
    """
    try:
        # First check if the file exists
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        
        # Extract file content
        content = load_document(file_path)
        
        if not content:
            return f"Error: Could not extract content from file: {file_path}"
        
        # Get title from metadata or filename
        title = "Unknown Title"
        if metadata and "title" in metadata:
            title = metadata["title"]
        else:
            # Extract title from filename (remove extension and replace underscores/hyphens)
            title = os.path.splitext(os.path.basename(file_path))[0]
            title = title.replace('_', ' ').replace('-', ' ')
        
        # Determine authors from metadata
        authors = "Unknown Authors"
        if metadata and "authors" in metadata:
            if isinstance(metadata["authors"], list):
                authors = ", ".join(metadata["authors"])
            else:
                authors = str(metadata["authors"])
        
        # Combine metadata with content
        return f"Title: {title}\n\nAuthors: {authors}\n\nFull Text:\n{content}"
    
    except Exception as e:
        return f"Error loading local paper: {str(e)}"

def _fetch_url_content(url: str, metadata: Dict[str, Any] = None) -> str:
    """
    Fetch content from a URL, handling both PDF files and webpages.
    
    Args:
        url: URL to fetch
        metadata: Optional metadata about the document
        
    Returns:
        Document content as text
    """
    try:
        # Make the request
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10, stream=True)
        response.raise_for_status()
        
        # Get the response content
        content_type = response.headers.get('Content-Type', '')
        is_html = 'text/html' in content_type.lower()
        is_pdf = 'application/pdf' in content_type.lower() or url.lower().endswith('.pdf')
        
        # Handle PDF content
        if is_pdf:
            try:
                # Create a temporary directory to store the PDF
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Generate a filename from the URL
                    filename = url.split('/')[-1]
                    if not filename.endswith('.pdf'):
                        filename = 'document.pdf'
                    
                    pdf_path = os.path.join(temp_dir, filename)
                    
                    # Download the PDF
                    with open(pdf_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Extract text from the PDF
                    pdf_text = load_document(pdf_path)
                    
                    if pdf_text:
                        title = metadata.get('title', filename) if metadata else filename
                        return f"Title: {title}\n\nContent:\n{pdf_text}"
            
            except Exception as pdf_error:
                print(f"Error processing PDF: {str(pdf_error)}")
                return f"Error processing PDF: {str(pdf_error)}"
        
        # Handle HTML content
        elif is_html:
            try:
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.text.strip() if title_tag else "Untitled webpage"
                
                # Extract text from body
                text = ""
                if soup.body:
                    # Remove script and style elements
                    for script in soup.select("script, style, footer, nav, header"):
                        script.extract()
                    
                    # Get text content
                    text = soup.body.get_text(separator='\n')
                    
                    # Clean up whitespace
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                    text = re.sub(r'[ \t]+', ' ', text)
                    text = text.strip()
                
                return f"Title: {title}\n\nContent:\n{text}"
            
            except Exception as html_error:
                print(f"Error processing HTML: {str(html_error)}")
                return f"Error processing HTML: {str(html_error)}"
        
        # For other content types, just return raw text
        else:
            return f"Content:\n{response.text}"
    
    except Exception as e:
        return f"Error fetching URL content: {str(e)}"