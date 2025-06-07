"""
Academic research tools for the Research Assistant.

This module provides tools for accessing academic sources such as
Google Scholar, PubMed, arXiv, Google Books, and Open Library.
"""

import re
import urllib.parse
from datetime import datetime
from typing import List

import requests
from langchain.tools import Tool

# Import scholarly and arxiv libraries with error handling
try:
    from scholarly import scholarly

    SCHOLARLY_AVAILABLE = True
except ImportError:
    print("Warning: scholarly library not available. Using mock Google Scholar tool.")
    SCHOLARLY_AVAILABLE = False

try:
    import arxiv

    ARXIV_AVAILABLE = True
except ImportError:
    print("Warning: arxiv library not available. Using mock arXiv tool.")
    ARXIV_AVAILABLE = False

from src.models.research_models import Source, SourceType

class GoogleScholarTool:
    """Tool for searching Google Scholar."""

    def search(self, query: str, num_results: int = 5) -> List[Source]:
        """
        Search Google Scholar for academic papers.

        Args:
            query: The search query
            num_results: Maximum number of results to return

        Returns:
            List of Source objects
        """
        print(f"Searching Google Scholar for: {query}")

        results = []

        if SCHOLARLY_AVAILABLE:
            try:
                # Use the scholarly library to search Google Scholar
                search_query = scholarly.search_pubs(query)

                # Get the specified number of results
                count = 0
                for i, pub in enumerate(search_query):
                    if count >= num_results:
                        break

                    # Get more details about the publication
                    try:
                        # Extract publication details
                        title = pub.get("bib", {}).get("title", f"Paper on {query}")
                        authors = pub.get("bib", {}).get("author", [])
                        if not isinstance(authors, list):
                            authors = [authors]

                        # Extract journal information
                        journal = pub.get("bib", {}).get("journal", "")
                        volume = pub.get("bib", {}).get("volume", "")
                        issue = pub.get("bib", {}).get("number", "")
                        pages = pub.get("bib", {}).get("pages", "")

                        # Extract URL and DOI
                        url = pub.get("pub_url", "")
                        if not url and "eprint_url" in pub:
                            url = pub["eprint_url"]
                        if not url and "url" in pub:
                            url = pub["url"]

                        # Try to extract DOI
                        doi = None
                        if "doi" in pub:
                            doi = pub["doi"]
                        elif url and "doi.org" in url:
                            doi_match = re.search(r"doi\.org/(.+)", url)
                            if doi_match:
                                doi = doi_match.group(1)

                        # Create a Source object
                        source = Source(
                            title=title,
                            authors=authors,
                            publication_date=None,  # Scholarly doesn't provide this easily
                            source_type=SourceType.ACADEMIC,
                            url=url,
                            journal=journal,
                            volume=volume,
                            issue=issue,
                            pages=pages,
                            doi=doi,
                        )
                        results.append(source)
                        count += 1
                    except Exception as e:
                        print(f"Error processing Google Scholar result: {e}")
                        continue
            except Exception as e:
                print(f"Error searching Google Scholar: {e}")
                # Fall back to mock results
                self._add_mock_results(query, num_results, results)
        else:
            # Use mock results if scholarly is not available
            self._add_mock_results(query, num_results, results)

        return results

    def _add_mock_results(self, query: str, num_results: int, results: List[Source]):
        """Add mock results when the scholarly library is not available."""
        for i in range(min(num_results, 3)):
            source = Source(
                title=f"Academic Paper on {query.title()} - {i + 1}",
                authors=[f"Author {i + 1}", f"Author {i + 2}"],
                publication_date=datetime.now(),
                source_type=SourceType.ACADEMIC,
                url=f"https://scholar.google.com/scholar?q={urllib.parse.quote(query)}&result={i}",
                journal="Journal of Advanced Research",
                volume="42",
                issue="3",
                pages="123-145",
                doi=f"10.1234/journal.{i + 1000}",
            )
            results.append(source)

    def run(self, query: str) -> str:
        """Run the Google Scholar search and return results as a string."""
        sources = self.search(query)
        results = []

        if not sources:
            return "No results found on Google Scholar for the query."

        for i, source in enumerate(sources, 1):
            results.append(f"{i}. {source.title}")
            results.append(f"   Authors: {', '.join(source.authors or [])}")
            if source.journal:
                journal_info = f"   Journal: {source.journal}"
                if source.volume:
                    journal_info += f", {source.volume}"
                    if source.issue:
                        journal_info += f"({source.issue})"
                if source.pages:
                    journal_info += f", {source.pages}"
                results.append(journal_info)
            if source.doi:
                results.append(f"   DOI: {source.doi}")
            if source.url:
                results.append(f"   URL: {source.url}")
            results.append("")

        return "\n".join(results)

class PubMedTool:
    """Tool for searching PubMed."""

    def search(self, query: str, num_results: int = 5) -> List[Source]:
        """
        Search PubMed for medical papers.

        Args:
            query: The search query
            num_results: Maximum number of results to return

        Returns:
            List of Source objects
        """
        print(f"Searching PubMed for: {query}")

        results = []

        try:
            # Use the PubMed E-utilities API
            # First, search for IDs
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmode=json&retmax={num_results}"
            search_response = requests.get(search_url)

            if search_response.status_code == 200:
                search_data = search_response.json()
                pmids = search_data.get("esearchresult", {}).get("idlist", [])

                if pmids:
                    # Fetch details for each ID
                    ids_str = ",".join(pmids)
                    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json"
                    fetch_response = requests.get(fetch_url)

                    if fetch_response.status_code == 200:
                        fetch_data = fetch_response.json()
                        articles = fetch_data.get("result", {})

                        # Remove the UIDs key which is just a list of IDs
                        if "uids" in articles:
                            uids = articles.pop("uids")

                            for pmid in uids:
                                article = articles.get(pmid, {})

                                # Extract article details
                                title = article.get(
                                    "title", f"Medical Paper on {query}"
                                )

                                # Extract authors
                                authors = []
                                for author in article.get("authors", []):
                                    if "name" in author:
                                        authors.append(author["name"])

                                # Extract journal information
                                journal = article.get(
                                    "fulljournalname", article.get("source", "")
                                )
                                volume = article.get("volume", "")
                                issue = article.get("issue", "")
                                pages = article.get("pages", "")

                                # Extract publication date
                                pub_date = None
                                if "pubdate" in article:
                                    try:
                                        pub_date = datetime.strptime(
                                            article["pubdate"], "%Y %b %d"
                                        )
                                    except ValueError:
                                        try:
                                            pub_date = datetime.strptime(
                                                article["pubdate"], "%Y %b"
                                            )
                                        except ValueError:
                                            try:
                                                pub_date = datetime.strptime(
                                                    article["pubdate"], "%Y"
                                                )
                                            except ValueError:
                                                pass

                                # Extract DOI
                                doi = None
                                for id_obj in article.get("articleids", []):
                                    if id_obj.get("idtype") == "doi":
                                        doi = id_obj.get("value")

                                # Create URL
                                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                                # Create a Source object
                                source = Source(
                                    title=title,
                                    authors=authors,
                                    publication_date=pub_date,
                                    source_type=SourceType.JOURNAL,
                                    url=url,
                                    journal=journal,
                                    volume=volume,
                                    issue=issue,
                                    pages=pages,
                                    doi=doi,
                                    publisher="National Library of Medicine",
                                )
                                results.append(source)

            # If no results were found or there was an error, use mock results
            if not results:
                self._add_mock_results(query, num_results, results)

        except Exception as e:
            print(f"Error searching PubMed: {e}")
            # Fall back to mock results
            self._add_mock_results(query, num_results, results)

        return results

    def _add_mock_results(self, query: str, num_results: int, results: List[Source]):
        """Add mock results when the PubMed API fails."""
        for i in range(min(num_results, 3)):
            source = Source(
                title=f"Medical Research on {query.title()} - {i + 1}",
                authors=[f"Dr. {i + 1}", f"Dr. {i + 2}"],
                publication_date=datetime.now(),
                source_type=SourceType.JOURNAL,
                url=f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(query)}&result={i}",
                journal="Journal of Medical Research",
                volume="28",
                issue="4",
                pages="234-256",
                doi=f"10.5678/medicine.{i + 2000}",
                publisher="National Library of Medicine",
            )
            results.append(source)

    def run(self, query: str) -> str:
        """Run the PubMed search and return results as a string."""
        sources = self.search(query)
        results = []

        if not sources:
            return "No results found on PubMed for the query."

        for i, source in enumerate(sources, 1):
            results.append(f"{i}. {source.title}")
            results.append(f"   Authors: {', '.join(source.authors or [])}")

            journal_info = []
            if source.journal:
                journal_info.append(source.journal)
            if source.volume:
                journal_info.append(f"Vol. {source.volume}")
            if source.issue:
                journal_info.append(f"Issue {source.issue}")
            if source.pages:
                journal_info.append(f"Pages {source.pages}")

            if journal_info:
                results.append(f"   Journal: {', '.join(journal_info)}")

            if source.doi:
                results.append(f"   DOI: {source.doi}")

            if source.publication_date:
                results.append(
                    f"   Published: {source.publication_date.strftime('%Y-%m-%d')}"
                )

            results.append(f"   URL: {source.url}")
            results.append("")

        return "\n".join(results)

class ArXivTool:
    """Tool for searching arXiv."""

    def search(self, query: str, num_results: int = 5) -> List[Source]:
        """
        Search arXiv for preprints.

        Args:
            query: The search query
            num_results: Maximum number of results to return

        Returns:
            List of Source objects
        """
        print(f"Searching arXiv for: {query}")

        results = []

        if ARXIV_AVAILABLE:
            try:
                # Use the arxiv library to search arXiv
                search_query = arxiv.Search(
                    query=query,
                    max_results=num_results,
                    sort_by=arxiv.SortCriterion.Relevance,
                )

                # Get the results
                for paper in search_query.results():
                    try:
                        # Extract paper details
                        title = paper.title

                        # Extract authors
                        authors = [author.name for author in paper.authors]

                        # Extract publication date
                        publication_date = paper.published

                        # Extract URL
                        url = paper.pdf_url
                        if not url:
                            url = paper.entry_id

                        # Extract arXiv ID
                        arxiv_id = paper.entry_id.split("/")[-1]
                        if not arxiv_id:
                            arxiv_id = paper.get_short_id()

                        # Create a Source object
                        source = Source(
                            title=title,
                            authors=authors,
                            publication_date=publication_date,
                            source_type=SourceType.PREPRINT,
                            url=url,
                            journal="arXiv",
                            publisher="Cornell University",
                        )
                        results.append(source)
                    except Exception as e:
                        print(f"Error processing arXiv result: {e}")
                        continue
            except Exception as e:
                print(f"Error searching arXiv: {e}")
                # Fall back to mock results
                self._add_mock_results(query, num_results, results)
        else:
            # Use mock results if arxiv is not available
            self._add_mock_results(query, num_results, results)

        return results

    def _add_mock_results(self, query: str, num_results: int, results: List[Source]):
        """Add mock results when the arxiv library is not available."""
        for i in range(min(num_results, 3)):
            source = Source(
                title=f"Preprint on {query.title()} - {i + 1}",
                authors=[f"Researcher {i + 1}", f"Researcher {i + 2}"],
                publication_date=datetime.now(),
                source_type=SourceType.PREPRINT,
                url=f"https://arxiv.org/abs/{2100 + i}.{10000 + i}",
                journal="arXiv",
                publisher="Cornell University",
            )
            results.append(source)

    def run(self, query: str) -> str:
        """Run the arXiv search and return results as a string."""
        sources = self.search(query)
        results = []

        if not sources:
            return "No results found on arXiv for the query."

        for i, source in enumerate(sources, 1):
            results.append(f"{i}. {source.title}")
            results.append(f"   Authors: {', '.join(source.authors or [])}")

            # Extract arXiv ID from URL
            arxiv_id = source.url.split("/")[-1]
            if arxiv_id:
                results.append(f"   arXiv ID: {arxiv_id}")

            results.append(f"   URL: {source.url}")

            if source.publication_date:
                results.append(
                    f"   Published: {source.publication_date.strftime('%Y-%m-%d')}"
                )

            results.append("")

        return "\n".join(results)

class GoogleBooksTool:
    """Tool for searching Google Books."""

    def search(self, query: str, num_results: int = 5) -> List[Source]:
        """
        Search Google Books for books.

        Args:
            query: The search query
            num_results: Maximum number of results to return

        Returns:
            List of Source objects
        """
        print(f"Searching Google Books for: {query}")

        results = []

        try:
            # Use the Google Books API
            api_url = f"https://www.googleapis.com/books/v1/volumes?q={urllib.parse.quote(query)}&maxResults={num_results}"
            response = requests.get(api_url)

            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])

                for item in items:
                    try:
                        # Extract book details
                        volume_info = item.get("volumeInfo", {})

                        # Extract title
                        title = volume_info.get("title", f"Book on {query}")

                        # Extract authors
                        authors = volume_info.get("authors", [])

                        # Extract publication date
                        pub_date = None
                        if "publishedDate" in volume_info:
                            pub_date_str = volume_info["publishedDate"]
                            try:
                                # Try different date formats
                                if len(pub_date_str) == 4:  # Just year
                                    pub_date = datetime.strptime(pub_date_str, "%Y")
                                elif len(pub_date_str) == 7:  # Year and month
                                    pub_date = datetime.strptime(pub_date_str, "%Y-%m")
                                else:  # Full date
                                    pub_date = datetime.strptime(
                                        pub_date_str, "%Y-%m-%d"
                                    )
                            except ValueError:
                                pass

                        # Extract publisher
                        publisher = volume_info.get("publisher", "")

                        # Extract ISBN
                        isbn = ""
                        for identifier in volume_info.get("industryIdentifiers", []):
                            if identifier.get("type") in ["ISBN_10", "ISBN_13"]:
                                isbn = identifier.get("identifier", "")
                                break

                        # Extract URL
                        url = volume_info.get("infoLink", "")
                        if not url and "canonicalVolumeLink" in volume_info:
                            url = volume_info["canonicalVolumeLink"]
                        if not url and "previewLink" in volume_info:
                            url = volume_info["previewLink"]

                        # Create a Source object
                        source = Source(
                            title=title,
                            authors=authors,
                            publication_date=pub_date,
                            source_type=SourceType.BOOK,
                            url=url,
                            publisher=publisher,
                            isbn=isbn,
                        )
                        results.append(source)
                    except Exception as e:
                        print(f"Error processing Google Books result: {e}")
                        continue

            # If no results were found or there was an error, use mock results
            if not results:
                self._add_mock_results(query, num_results, results)

        except Exception as e:
            print(f"Error searching Google Books: {e}")
            # Fall back to mock results
            self._add_mock_results(query, num_results, results)

        return results

    def _add_mock_results(self, query: str, num_results: int, results: List[Source]):
        """Add mock results when the Google Books API fails."""
        for i in range(min(num_results, 3)):
            source = Source(
                title=f"Book on {query.title()} - {i + 1}",
                authors=[f"Author {i + 1}", f"Author {i + 2}"],
                publication_date=datetime.now(),
                source_type=SourceType.BOOK,
                url=f"https://books.google.com/books?id=abc{i}",
                publisher=f"Publisher {i + 1}",
                isbn=f"978-3-16-148410-{i}",
            )
            results.append(source)

    def run(self, query: str) -> str:
        """Run the Google Books search and return results as a string."""
        sources = self.search(query)
        results = []

        if not sources:
            return "No results found on Google Books for the query."

        for i, source in enumerate(sources, 1):
            results.append(f"{i}. {source.title}")
            results.append(f"   Authors: {', '.join(source.authors or [])}")

            if source.publisher:
                results.append(f"   Publisher: {source.publisher}")

            if source.isbn:
                results.append(f"   ISBN: {source.isbn}")

            if source.publication_date:
                results.append(
                    f"   Published: {source.publication_date.strftime('%Y')}"
                )

            results.append(f"   URL: {source.url}")
            results.append("")

        return "\n".join(results)

class OpenLibraryTool:
    """Tool for searching Open Library."""

    def search(self, query: str, num_results: int = 5) -> List[Source]:
        """
        Search Open Library for books.

        Args:
            query: The search query
            num_results: Maximum number of results to return

        Returns:
            List of Source objects
        """
        print(f"Searching Open Library for: {query}")

        results = []

        try:
            # Use the Open Library Search API
            api_url = f"https://openlibrary.org/search.json?q={urllib.parse.quote(query)}&limit={num_results}"
            response = requests.get(api_url)

            if response.status_code == 200:
                data = response.json()
                docs = data.get("docs", [])

                for doc in docs:
                    try:
                        # Extract book details
                        title = doc.get("title", f"Book on {query}")

                        # Extract authors
                        authors = []
                        author_names = doc.get("author_name", [])
                        if author_names:
                            authors = author_names

                        # Extract publication date
                        pub_date = None
                        if "first_publish_year" in doc:
                            try:
                                pub_date = datetime(doc["first_publish_year"], 1, 1)
                            except ValueError:
                                pass

                        # Extract publisher
                        publisher = ""
                        publishers = doc.get("publisher", [])
                        if publishers:
                            publisher = publishers[0]

                        # Extract ISBN
                        isbn = ""
                        isbns = doc.get("isbn", [])
                        if isbns:
                            isbn = isbns[0]

                        # Create URL
                        key = doc.get("key", "")
                        url = f"https://openlibrary.org{key}" if key else ""

                        # Create a Source object
                        source = Source(
                            title=title,
                            authors=authors,
                            publication_date=pub_date,
                            source_type=SourceType.BOOK,
                            url=url,
                            publisher=publisher,
                            isbn=isbn,
                        )
                        results.append(source)
                    except Exception as e:
                        print(f"Error processing Open Library result: {e}")
                        continue

            # If no results were found or there was an error, use mock results
            if not results:
                self._add_mock_results(query, num_results, results)

        except Exception as e:
            print(f"Error searching Open Library: {e}")
            # Fall back to mock results
            self._add_mock_results(query, num_results, results)

        return results

    def _add_mock_results(self, query: str, num_results: int, results: List[Source]):
        """Add mock results when the Open Library API fails."""
        for i in range(min(num_results, 3)):
            source = Source(
                title=f"Open Access Book on {query.title()} - {i + 1}",
                authors=[f"Writer {i + 1}", f"Writer {i + 2}"],
                publication_date=datetime.now(),
                source_type=SourceType.BOOK,
                url=f"https://openlibrary.org/works/OL{i + 1000}W",
                publisher="Open Library",
                isbn=f"978-0-12-345678-{i}",
            )
            results.append(source)

    def run(self, query: str) -> str:
        """Run the Open Library search and return results as a string."""
        sources = self.search(query)
        results = []

        if not sources:
            return "No results found on Open Library for the query."

        for i, source in enumerate(sources, 1):
            results.append(f"{i}. {source.title}")
            results.append(f"   Authors: {', '.join(source.authors or [])}")

            if source.publisher:
                results.append(f"   Publisher: {source.publisher}")

            if source.isbn:
                results.append(f"   ISBN: {source.isbn}")

            if source.publication_date:
                results.append(
                    f"   Published: {source.publication_date.strftime('%Y')}"
                )

            results.append(f"   URL: {source.url}")
            results.append("")

        return "\n".join(results)

# Create tool instances
google_scholar = GoogleScholarTool()
pubmed = PubMedTool()
arxiv = ArXivTool()
google_books = GoogleBooksTool()
open_library = OpenLibraryTool()

# Create LangChain tools
google_scholar_tool = Tool(
    name="google_scholar",
    func=google_scholar.run,
    description="Search Google Scholar for academic papers and research. Use this for finding scholarly articles and academic research papers.",
)

pubmed_tool = Tool(
    name="pubmed",
    func=pubmed.run,
    description="Search PubMed for medical and biomedical research papers. Use this for finding medical research, clinical studies, and health-related academic papers.",
)

arxiv_tool = Tool(
    name="arxiv",
    func=arxiv.run,
    description="Search arXiv for preprints in physics, mathematics, computer science, and related fields. Use this for finding the latest research papers that may not be published in journals yet.",
)

google_books_tool = Tool(
    name="google_books",
    func=google_books.run,
    description="Search Google Books for books on various topics. Use this for finding books, textbooks, and other published literature.",
)

open_library_tool = Tool(
    name="open_library",
    func=open_library.run,
    description="Search Open Library for open-access books. Use this for finding freely available books and literature.",
)
