"""
Utilities for synthesis service including export functionality and HTML generation.

This module provides helper functions for exporting synthesis results in various
formats and generating HTML views of synthesized content.
"""

import json
import csv
import io
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from app.core.logging import get_logger
from app.models.synthesis_schemas import SynthesisResponse, OutputFormat, CitationStyle

logger = get_logger(__name__)


class ExportManager:
    """Handles exporting synthesis results to various formats."""
    
    def __init__(self):
        """Initialize export manager."""
        pass
    
    def export_to_json(self, synthesis_response: Dict[str, Any]) -> str:
        """
        Export synthesis response to JSON format.
        
        Args:
            synthesis_response: Complete synthesis response
            
        Returns:
            JSON string representation
        """
        try:
            return json.dumps(synthesis_response, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return json.dumps({"error": f"Export failed: {e}"})
    
    def export_to_markdown(self, synthesis_response: Dict[str, Any]) -> str:
        """
        Export synthesis response to markdown format.
        
        Args:
            synthesis_response: Complete synthesis response
            
        Returns:
            Markdown representation
        """
        try:
            md_content = []
            
            # Add metadata header
            md_content.append(f"# Synthesis Report")
            md_content.append(f"**Query:** {synthesis_response.get('query', 'Unknown')}")
            md_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            md_content.append(f"**Format:** {synthesis_response.get('format', 'Unknown')}")
            md_content.append(f"**Style:** {synthesis_response.get('style', 'Unknown')}")
            md_content.append("")
            
            # Add main content
            answer_markdown = synthesis_response.get('answer_markdown', '')
            if answer_markdown:
                md_content.append("## Content")
                md_content.append(answer_markdown)
                md_content.append("")
            
            # Add sources if available
            sources = synthesis_response.get('sources', [])
            if sources:
                md_content.append("## Sources")
                for i, source in enumerate(sources, 1):
                    source_text = source.get('formatted_citation', f'Source {i}')
                    md_content.append(f"{i}. {source_text}")
                md_content.append("")
            
            # Add warnings if any
            warnings = synthesis_response.get('warnings', [])
            if warnings:
                md_content.append("## Warnings")
                for warning in warnings:
                    md_content.append(f"- {warning}")
                md_content.append("")
            
            # Add metadata
            metadata = synthesis_response.get('metadata', {})
            if metadata:
                md_content.append("## Processing Information")
                processing_time = metadata.get('retrieval_time_ms', 0) + metadata.get('llm_time_ms', 0)
                md_content.append(f"- **Total Processing Time:** {processing_time}ms")
                md_content.append(f"- **Documents Retrieved:** {len(sources)}")
                md_content.append(f"- **Tokens Used:** {synthesis_response.get('tokens_used', 'Unknown')}")
            
            return "\n".join(md_content)
            
        except Exception as e:
            logger.error(f"Error exporting to Markdown: {e}")
            return f"# Export Error\n\nFailed to export synthesis: {e}"
    
    def export_to_csv(self, synthesis_response: Dict[str, Any]) -> str:
        """
        Export synthesis response to CSV format (for data analysis).
        
        Args:
            synthesis_response: Complete synthesis response
            
        Returns:
            CSV string representation
        """
        try:
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers
            writer.writerow([
                'Query', 'Format', 'Style', 'Answer_Text', 'Sources_Count',
                'Processing_Time_MS', 'Tokens_Used', 'Quality_Score', 'Warnings_Count'
            ])
            
            # Extract data
            query = synthesis_response.get('query', '')
            format_val = synthesis_response.get('format', '')
            style = synthesis_response.get('style', '')
            answer_text = synthesis_response.get('answer_text', '').replace('\n', ' ')[:500]  # Truncate for CSV
            sources_count = len(synthesis_response.get('sources', []))
            processing_time = synthesis_response.get('processing_time_ms', 0)
            tokens_used = synthesis_response.get('tokens_used', 0)
            
            # Calculate quality score (simplified)
            metadata = synthesis_response.get('metadata', {})
            quality_check = metadata.get('quality_check', {})
            quality_score = quality_check.get('overall_quality', 'Unknown')
            
            warnings_count = len(synthesis_response.get('warnings', []))
            
            # Write data row
            writer.writerow([
                query, format_val, style, answer_text, sources_count,
                processing_time, tokens_used, quality_score, warnings_count
            ])
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return f"Error,Export failed: {e}\n"
    
    def export_citations_only(
        self, 
        synthesis_response: Dict[str, Any],
        citation_style: CitationStyle = CitationStyle.APA
    ) -> str:
        """
        Export only the citations in specified format.
        
        Args:
            synthesis_response: Complete synthesis response
            citation_style: Citation format to use
            
        Returns:
            Formatted citations string
        """
        try:
            sources = synthesis_response.get('sources', [])
            if not sources:
                return "No sources available."
            
            citations = []
            citations.append(f"# References ({citation_style.value} Style)")
            citations.append("")
            
            for i, source in enumerate(sources, 1):
                citation = source.get('formatted_citation', f'Source {i}')
                citations.append(f"{i}. {citation}")
            
            return "\n".join(citations)
            
        except Exception as e:
            logger.error(f"Error exporting citations: {e}")
            return f"Error exporting citations: {e}"


class HTMLGenerator:
    """Generates HTML views for synthesis results."""
    
    def __init__(self):
        """Initialize HTML generator."""
        self.base_css = self._get_base_css()
    
    def generate_html_report(self, synthesis_response: Dict[str, Any]) -> str:
        """
        Generate complete HTML report from synthesis response.
        
        Args:
            synthesis_response: Complete synthesis response
            
        Returns:
            HTML string
        """
        try:
            html_parts = []
            
            # HTML document start
            html_parts.append("<!DOCTYPE html>")
            html_parts.append("<html lang='en'>")
            html_parts.append("<head>")
            html_parts.append("<meta charset='UTF-8'>")
            html_parts.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
            html_parts.append(f"<title>Synthesis Report - {synthesis_response.get('query', 'Unknown')}</title>")
            html_parts.append(f"<style>{self.base_css}</style>")
            html_parts.append("</head>")
            html_parts.append("<body>")
            
            # Header
            html_parts.append("<div class='header'>")
            html_parts.append("<h1>Synthesis Report</h1>")
            html_parts.append(f"<div class='meta-info'>")
            html_parts.append(f"<p><strong>Query:</strong> {synthesis_response.get('query', 'Unknown')}</p>")
            html_parts.append(f"<p><strong>Format:</strong> {synthesis_response.get('format', 'Unknown')}</p>")
            html_parts.append(f"<p><strong>Style:</strong> {synthesis_response.get('style', 'Unknown')}</p>")
            html_parts.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            html_parts.append("</div>")
            html_parts.append("</div>")
            
            # Main content
            html_parts.append("<div class='main-content'>")
            answer_html = synthesis_response.get('answer_html')
            if answer_html:
                html_parts.append(answer_html)
            else:
                # Convert markdown to basic HTML
                answer_markdown = synthesis_response.get('answer_markdown', '')
                html_content = self._markdown_to_html(answer_markdown)
                html_parts.append(html_content)
            html_parts.append("</div>")
            
            # Sidebar with metadata
            html_parts.append("<div class='sidebar'>")
            
            # Sources
            sources = synthesis_response.get('sources', [])
            if sources:
                html_parts.append("<div class='sources-section'>")
                html_parts.append("<h3>Sources</h3>")
                html_parts.append("<ol>")
                for source in sources:
                    citation = source.get('formatted_citation', 'Unknown source')
                    html_parts.append(f"<li>{citation}</li>")
                html_parts.append("</ol>")
                html_parts.append("</div>")
            
            # Warnings
            warnings = synthesis_response.get('warnings', [])
            if warnings:
                html_parts.append("<div class='warnings-section'>")
                html_parts.append("<h3>Warnings</h3>")
                html_parts.append("<ul>")
                for warning in warnings:
                    html_parts.append(f"<li class='warning'>{warning}</li>")
                html_parts.append("</ul>")
                html_parts.append("</div>")
            
            # Processing info
            metadata = synthesis_response.get('metadata', {})
            if metadata:
                html_parts.append("<div class='processing-info'>")
                html_parts.append("<h3>Processing Information</h3>")
                html_parts.append("<ul>")
                
                processing_time = synthesis_response.get('processing_time_ms', 0)
                html_parts.append(f"<li>Processing Time: {processing_time}ms</li>")
                
                tokens_used = synthesis_response.get('tokens_used', 0)
                html_parts.append(f"<li>Tokens Used: {tokens_used}</li>")
                
                html_parts.append(f"<li>Documents Retrieved: {len(sources)}</li>")
                
                quality_check = metadata.get('quality_check', {})
                if quality_check:
                    quality_score = quality_check.get('overall_quality', 'Unknown')
                    html_parts.append(f"<li>Quality Score: {quality_score}</li>")
                
                html_parts.append("</ul>")
                html_parts.append("</div>")
            
            html_parts.append("</div>")  # End sidebar
            
            # Footer
            html_parts.append("<div class='footer'>")
            html_parts.append("<p>Generated by Researcher Agent Synthesis System</p>")
            html_parts.append("</div>")
            
            html_parts.append("</body>")
            html_parts.append("</html>")
            
            return "\n".join(html_parts)
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return self._generate_error_html(str(e))
    
    def generate_print_friendly_html(self, synthesis_response: Dict[str, Any]) -> str:
        """
        Generate print-friendly HTML version.
        
        Args:
            synthesis_response: Complete synthesis response
            
        Returns:
            Print-optimized HTML string
        """
        # Simplified version for printing
        try:
            html_parts = []
            
            html_parts.append("<!DOCTYPE html>")
            html_parts.append("<html>")
            html_parts.append("<head>")
            html_parts.append("<meta charset='UTF-8'>")
            html_parts.append(f"<title>Synthesis Report - {synthesis_response.get('query', 'Unknown')}</title>")
            html_parts.append("<style>")
            html_parts.append(self._get_print_css())
            html_parts.append("</style>")
            html_parts.append("</head>")
            html_parts.append("<body>")
            
            # Simple header
            html_parts.append(f"<h1>Synthesis Report</h1>")
            html_parts.append(f"<p><strong>Query:</strong> {synthesis_response.get('query', 'Unknown')}</p>")
            html_parts.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            html_parts.append("<hr>")
            
            # Main content
            answer_markdown = synthesis_response.get('answer_markdown', '')
            if answer_markdown:
                html_content = self._markdown_to_html(answer_markdown)
                html_parts.append(html_content)
            
            # Sources
            sources = synthesis_response.get('sources', [])
            if sources:
                html_parts.append("<h2>References</h2>")
                html_parts.append("<ol>")
                for source in sources:
                    citation = source.get('formatted_citation', 'Unknown source')
                    html_parts.append(f"<li>{citation}</li>")
                html_parts.append("</ol>")
            
            html_parts.append("</body>")
            html_parts.append("</html>")
            
            return "\n".join(html_parts)
            
        except Exception as e:
            logger.error(f"Error generating print-friendly HTML: {e}")
            return self._generate_error_html(str(e))
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """
        Convert markdown to basic HTML (simple implementation).
        
        Args:
            markdown_text: Markdown content
            
        Returns:
            HTML content
        """
        if not markdown_text:
            return ""
        
        try:
            # Try to use markdown library if available
            try:
                import markdown
                return markdown.markdown(markdown_text)
            except ImportError:
                pass
            
            # Simple fallback conversion
            html = markdown_text
            
            # Headers
            html = re.sub(r'^### (.*$)', r'<h3>\1</h3>', html, flags=re.MULTILINE)
            html = re.sub(r'^## (.*$)', r'<h2>\1</h2>', html, flags=re.MULTILINE)
            html = re.sub(r'^# (.*$)', r'<h1>\1</h1>', html, flags=re.MULTILINE)
            
            # Bold and italic
            html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html)
            html = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', html)
            
            # Links
            html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
            
            # Lists (simple)
            html = re.sub(r'^- (.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)
            html = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', html, flags=re.DOTALL)
            
            # Paragraphs
            paragraphs = html.split('\n\n')
            html_paragraphs = []
            for para in paragraphs:
                para = para.strip()
                if para and not para.startswith('<'):
                    html_paragraphs.append(f'<p>{para}</p>')
                else:
                    html_paragraphs.append(para)
            
            return '\n'.join(html_paragraphs)
            
        except Exception as e:
            logger.warning(f"Error converting markdown to HTML: {e}")
            return f'<pre>{markdown_text}</pre>'
    
    def _generate_error_html(self, error_message: str) -> str:
        """Generate error HTML page."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Synthesis Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .error {{ color: red; background: #ffe6e6; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Synthesis Report Error</h1>
            <div class="error">
                <h2>Error Generating Report</h2>
                <p>{error_message}</p>
            </div>
        </body>
        </html>
        """
    
    def _get_base_css(self) -> str:
        """Get base CSS for HTML reports."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            display: grid;
            grid-template-columns: 1fr 300px;
            grid-template-rows: auto 1fr auto;
            grid-template-areas: 
                "header header"
                "main sidebar"
                "footer footer";
            min-height: 100vh;
            gap: 20px;
        }
        
        .header {
            grid-area: header;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            margin: 0 0 15px 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .meta-info p {
            margin: 5px 0;
            opacity: 0.9;
        }
        
        .main-content {
            grid-area: main;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-left: 20px;
        }
        
        .sidebar {
            grid-area: sidebar;
            padding: 20px;
            margin-right: 20px;
        }
        
        .sources-section, .warnings-section, .processing-info {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .sources-section h3, .warnings-section h3, .processing-info h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .warning {
            color: #d4691e;
            font-weight: 500;
        }
        
        .footer {
            grid-area: footer;
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        h1, h2, h3 {
            color: #333;
        }
        
        h2 {
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        blockquote {
            border-left: 4px solid #667eea;
            padding-left: 20px;
            margin-left: 0;
            font-style: italic;
            background: #f8f9fa;
            padding: 15px 15px 15px 25px;
            border-radius: 0 5px 5px 0;
        }
        
        code {
            background: #f1f3f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #667eea;
        }
        
        @media (max-width: 768px) {
            body {
                grid-template-columns: 1fr;
                grid-template-areas: 
                    "header"
                    "main"
                    "sidebar"
                    "footer";
            }
            
            .main-content, .sidebar {
                margin: 0 10px;
            }
        }
        """
    
    def _get_print_css(self) -> str:
        """Get CSS optimized for printing."""
        return """
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: black;
            background: white;
        }
        
        h1, h2, h3 {
            color: black;
            page-break-after: avoid;
        }
        
        h1 {
            font-size: 24pt;
            margin-bottom: 20pt;
        }
        
        h2 {
            font-size: 18pt;
            margin-top: 20pt;
            margin-bottom: 10pt;
        }
        
        h3 {
            font-size: 14pt;
            margin-top: 15pt;
            margin-bottom: 8pt;
        }
        
        p {
            margin-bottom: 10pt;
            text-align: justify;
        }
        
        ol, ul {
            margin-bottom: 10pt;
        }
        
        li {
            margin-bottom: 5pt;
        }
        
        hr {
            border: none;
            border-top: 1px solid #ccc;
            margin: 20pt 0;
        }
        
        @media print {
            body {
                margin: 0;
                padding: 15mm;
            }
            
            h1, h2, h3 {
                page-break-after: avoid;
            }
            
            p, li {
                page-break-inside: avoid;
            }
        }
        """


# Global instances
export_manager = ExportManager()
html_generator = HTMLGenerator()


def get_export_manager() -> ExportManager:
    """Get the global export manager instance."""
    return export_manager


def get_html_generator() -> HTMLGenerator:
    """Get the global HTML generator instance."""
    return html_generator
