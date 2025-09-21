"""
Citation management system for the synthesis module.

Handles generation of citations in multiple formats (APA, MLA, Chicago),
tracks source usage, and manages in-text citations and reference lists.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from app.core.logging import get_logger
from app.models.synthesis_schemas import (
    Citation, CitationStyle, SourceMetadata, SupportingSource
)

logger = get_logger(__name__)


class CitationFormatter:
    """Formats citations according to different academic styles."""
    
    def __init__(self):
        """Initialize citation formatter."""
        self.citation_counter = defaultdict(int)
        self.used_sources = {}  # doc_id -> Citation
        
    def format_citation(
        self, 
        metadata: SourceMetadata, 
        style: CitationStyle
    ) -> Citation:
        """
        Format a citation according to the specified style.
        
        Args:
            metadata: Source metadata
            style: Citation style (APA, MLA, Chicago)
            
        Returns:
            Formatted citation object
        """
        try:
            if style == CitationStyle.APA:
                return self._format_apa(metadata)
            elif style == CitationStyle.MLA:
                return self._format_mla(metadata)
            elif style == CitationStyle.CHICAGO:
                return self._format_chicago(metadata)
            else:
                raise ValueError(f"Unsupported citation style: {style}")
                
        except Exception as e:
            logger.warning(f"Error formatting citation: {e}")
            # Return basic citation as fallback
            return Citation(
                source_metadata=metadata,
                citation_text=f"Unknown source ({metadata.doc_id})",
                in_text_citation=f"({metadata.doc_id})",
                style=style
            )
    
    def _format_apa(self, metadata: SourceMetadata) -> Citation:
        """Format citation in APA style."""
        # Extract year from date
        year = self._extract_year(metadata.date) or "n.d."
        
        # Author handling
        author = metadata.author or "Unknown author"
        if author != "Unknown author":
            # Format: Last, F. M.
            author_formatted = self._format_author_apa(author)
        else:
            author_formatted = author
        
        # Title handling
        title = metadata.title or f"Document {metadata.doc_id}"
        if metadata.url:
            title = f"{title}. Retrieved from {metadata.url}"
        
        # Full citation
        citation_parts = [author_formatted]
        citation_parts.append(f"({year}).")
        citation_parts.append(f"{title}.")
        
        if metadata.source and metadata.source != metadata.url:
            citation_parts.append(f"Source: {metadata.source}.")
        
        citation_text = " ".join(citation_parts)
        
        # In-text citation
        author_short = author.split(',')[0] if ',' in author else author.split()[0]
        in_text = f"({author_short}, {year})"
        
        return Citation(
            source_metadata=metadata,
            citation_text=citation_text,
            in_text_citation=in_text,
            style=CitationStyle.APA
        )
    
    def _format_mla(self, metadata: SourceMetadata) -> Citation:
        """Format citation in MLA style."""
        # Author handling
        author = metadata.author or "Unknown Author"
        if author != "Unknown Author":
            author_formatted = self._format_author_mla(author)
        else:
            author_formatted = author
        
        # Title in quotes
        title = metadata.title or f"Document {metadata.doc_id}"
        title = f'"{title}"'
        
        # Source and date
        source = metadata.source or "Web"
        date = self._format_date_mla(metadata.date)
        
        # Full citation
        citation_parts = [f"{author_formatted}."]
        citation_parts.append(f"{title}")
        citation_parts.append(f"{source},")
        if date:
            citation_parts.append(f"{date}.")
        
        if metadata.url:
            citation_parts.append(f"Web. <{metadata.url}>.")
        
        citation_text = " ".join(citation_parts)
        
        # In-text citation
        author_short = author.split(',')[0] if ',' in author else author.split()[0]
        in_text = f"({author_short})"
        
        return Citation(
            source_metadata=metadata,
            citation_text=citation_text,
            in_text_citation=in_text,
            style=CitationStyle.MLA
        )
    
    def _format_chicago(self, metadata: SourceMetadata) -> Citation:
        """Format citation in Chicago style."""
        # Author handling
        author = metadata.author or "Unknown Author"
        
        # Title handling
        title = metadata.title or f"Document {metadata.doc_id}"
        
        # Date handling
        year = self._extract_year(metadata.date) or "n.d."
        
        # Full citation (Notes-Bibliography style)
        citation_parts = [f"{author}."]
        citation_parts.append(f'"{title}."')
        
        if metadata.source:
            citation_parts.append(f"{metadata.source}.")
        
        if metadata.url:
            citation_parts.append(f"Accessed {datetime.now().strftime('%B %d, %Y')}. {metadata.url}.")
        
        citation_text = " ".join(citation_parts)
        
        # In-text citation (short form)
        author_short = author.split(',')[0] if ',' in author else author.split()[0]
        in_text = f"({author_short}, {year})"
        
        return Citation(
            source_metadata=metadata,
            citation_text=citation_text,
            in_text_citation=in_text,
            style=CitationStyle.CHICAGO
        )
    
    def _extract_year(self, date_str: Optional[str]) -> Optional[str]:
        """Extract year from date string."""
        if not date_str:
            return None
        
        # Try to find 4-digit year
        year_match = re.search(r'\b(\d{4})\b', date_str)
        if year_match:
            return year_match.group(1)
        
        return None
    
    def _format_author_apa(self, author: str) -> str:
        """Format author name for APA style."""
        if ',' in author:
            return author  # Assume already formatted
        
        # Split and reformat
        parts = author.strip().split()
        if len(parts) >= 2:
            last_name = parts[-1]
            first_names = ' '.join(parts[:-1])
            # Get initials
            initials = '. '.join([name[0] for name in first_names.split() if name]) + '.'
            return f"{last_name}, {initials}"
        
        return author
    
    def _format_author_mla(self, author: str) -> str:
        """Format author name for MLA style."""
        if ',' in author:
            return author  # Assume already formatted
        
        # Split and reformat
        parts = author.strip().split()
        if len(parts) >= 2:
            last_name = parts[-1]
            first_names = ' '.join(parts[:-1])
            return f"{last_name}, {first_names}"
        
        return author
    
    def _format_date_mla(self, date_str: Optional[str]) -> Optional[str]:
        """Format date for MLA style."""
        if not date_str:
            return None
        
        # Try to parse and reformat
        try:
            # Handle various date formats
            if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                return date_obj.strftime('%d %b %Y')
            elif re.match(r'\d{4}', date_str):
                return date_str  # Just year
        except:
            pass
        
        return date_str


class CitationManager:
    """
    Manages citations throughout the synthesis process.
    
    Tracks source usage, generates in-text citations, and creates
    properly formatted reference lists.
    """
    
    def __init__(self, style: CitationStyle = CitationStyle.APA):
        """Initialize citation manager."""
        self.style = style
        self.formatter = CitationFormatter()
        self.citations = {}  # doc_id -> Citation
        self.usage_count = defaultdict(int)
        self.in_text_citations = []  # Track in-text citation usage
        
    def add_source(self, metadata: SourceMetadata) -> Citation:
        """
        Add a source and generate its citation.
        
        Args:
            metadata: Source metadata
            
        Returns:
            Generated citation object
        """
        doc_id = metadata.doc_id
        
        if doc_id not in self.citations:
            citation = self.formatter.format_citation(metadata, self.style)
            self.citations[doc_id] = citation
            logger.debug(f"Added citation for {doc_id}")
        
        self.usage_count[doc_id] += 1
        return self.citations[doc_id]
    
    def get_in_text_citation(self, doc_id: str) -> str:
        """Get in-text citation for a document."""
        if doc_id in self.citations:
            citation = self.citations[doc_id].in_text_citation
            self.in_text_citations.append(citation)
            return citation
        
        return f"(Unknown source: {doc_id})"
    
    def generate_reference_list(self) -> str:
        """Generate formatted reference list."""
        if not self.citations:
            return "No sources cited."
        
        # Sort citations alphabetically by author
        sorted_citations = sorted(
            self.citations.values(),
            key=lambda c: c.source_metadata.author or c.source_metadata.doc_id
        )
        
        reference_lines = []
        if self.style == CitationStyle.APA:
            reference_lines.append("## References\n")
        elif self.style == CitationStyle.MLA:
            reference_lines.append("## Works Cited\n")
        elif self.style == CitationStyle.CHICAGO:
            reference_lines.append("## Bibliography\n")
        
        for citation in sorted_citations:
            reference_lines.append(f"{citation.citation_text}\n")
        
        return "\n".join(reference_lines)
    
    def generate_citations_json(self) -> List[Dict[str, Any]]:
        """Generate citations in JSON format for API responses."""
        citations_list = []
        
        for doc_id, citation in self.citations.items():
            citations_list.append({
                "doc_id": doc_id,
                "citation_text": citation.citation_text,
                "in_text_citation": citation.in_text_citation,
                "style": citation.style.value,
                "usage_count": self.usage_count[doc_id],
                "metadata": citation.source_metadata.dict()
            })
        
        return citations_list
    
    def validate_citations(self) -> List[str]:
        """Validate citations and return any warnings."""
        warnings = []
        
        for doc_id, citation in self.citations.items():
            metadata = citation.source_metadata
            
            # Check for missing essential information
            if not metadata.author and not metadata.title:
                warnings.append(f"Source {doc_id} lacks both author and title")
            
            if not metadata.date:
                warnings.append(f"Source {doc_id} lacks publication date")
            
            if not metadata.source and not metadata.url:
                warnings.append(f"Source {doc_id} lacks source information")
            
            # Check for unused sources
            if self.usage_count[doc_id] == 0:
                warnings.append(f"Source {doc_id} was added but never cited")
        
        return warnings
    
    def get_citation_stats(self) -> Dict[str, Any]:
        """Get statistics about citation usage."""
        return {
            "total_sources": len(self.citations),
            "citation_style": self.style.value,
            "total_in_text_citations": len(self.in_text_citations),
            "most_cited_source": max(self.usage_count.items(), key=lambda x: x[1]) if self.usage_count else None,
            "average_citations_per_source": sum(self.usage_count.values()) / len(self.citations) if self.citations else 0,
            "sources_by_type": self._get_source_type_distribution()
        }
    
    def _get_source_type_distribution(self) -> Dict[str, int]:
        """Get distribution of source types."""
        distribution = defaultdict(int)
        
        for citation in self.citations.values():
            metadata = citation.source_metadata
            
            # Classify source type based on URL or source info
            if metadata.url:
                if any(domain in metadata.url for domain in ['arxiv.org', 'ieee.org', 'acm.org']):
                    distribution['academic'] += 1
                elif any(domain in metadata.url for domain in ['.edu', '.gov']):
                    distribution['institutional'] += 1
                elif any(domain in metadata.url for domain in ['wikipedia.org', 'github.com']):
                    distribution['reference'] += 1
                else:
                    distribution['web'] += 1
            else:
                distribution['unknown'] += 1
        
        return dict(distribution)
    
    def clear(self):
        """Clear all citations and start fresh."""
        self.citations.clear()
        self.usage_count.clear()
        self.in_text_citations.clear()
        logger.debug("Citation manager cleared")


# Global citation manager instances for different styles
citation_managers = {
    CitationStyle.APA: CitationManager(CitationStyle.APA),
    CitationStyle.MLA: CitationManager(CitationStyle.MLA),
    CitationStyle.CHICAGO: CitationManager(CitationStyle.CHICAGO)
}


def get_citation_manager(style: CitationStyle) -> CitationManager:
    """Get citation manager for specified style."""
    return citation_managers[style]
