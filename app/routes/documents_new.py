"""Document management API routes."""
import asyncio
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends, Query
from fastapi.responses import JSONResponse

from app.core.exceptions import DocumentError, ProcessingError
from app.models.schemas import (
    Document,
    DocumentListResponse,
    DocumentListItem,
    DocumentStatusResponse,
    ProcessingStatus,
    DocumentType,
    DocumentMetadata,
    StandardResponse
)
from app.services.document_processor import DocumentProcessor
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory storage for demonstration (replace with database in production)
documents_store = {}


def get_document_processor() -> DocumentProcessor:
    """Dependency to get document processor."""
    return DocumentProcessor()


async def save_uploaded_file(upload_file: UploadFile, destination: Path) -> None:
    """Save uploaded file to destination."""
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file {upload_file.filename}: {str(e)}")
        raise DocumentError(f"Failed to save file: {str(e)}")


def get_file_type(filename: str) -> DocumentType:
    """Determine file type from filename."""
    extension = Path(filename).suffix.lower()
    type_mapping = {
        '.pdf': DocumentType.PDF,
        '.docx': DocumentType.DOCX,
        '.doc': DocumentType.DOCX,
        '.txt': DocumentType.TXT,
        '.md': DocumentType.MARKDOWN,
        '.markdown': DocumentType.MARKDOWN
    }
    return type_mapping.get(extension, DocumentType.TXT)


@router.post("/upload", response_model=StandardResponse)
async def upload_document(
    file: UploadFile = File(...),
    processor: DocumentProcessor = Depends(get_document_processor)
) -> StandardResponse:
    """
    Upload and process a document.
    
    Supports PDF, DOCX, TXT, and Markdown files.
    """
    try:
        # Validate file
        if not file.filename:
            raise DocumentError("No filename provided")
        
        # Check file type
        file_type = get_file_type(file.filename)
        allowed_types = {DocumentType.PDF, DocumentType.DOCX, DocumentType.TXT, DocumentType.MARKDOWN}
        if file_type not in allowed_types:
            raise DocumentError(f"Unsupported file type: {Path(file.filename).suffix}")
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = uploads_dir / f"{document_id}_{file.filename}"
        await save_uploaded_file(file, file_path)
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Create initial document metadata
        metadata = DocumentMetadata(
            filename=file.filename,
            file_size=file_size,
            file_type=file_type,
            mime_type=file.content_type or "application/octet-stream"
        )
        
        # Create document record
        document = Document(
            id=document_id,
            filename=file.filename,
            file_path=str(file_path),
            metadata=metadata,
            processing_status=ProcessingStatus.PENDING,
            upload_timestamp=datetime.utcnow()
        )
        
        # Store document
        documents_store[document_id] = document
        
        # Start background processing
        asyncio.create_task(process_document_async(document_id, processor))
        
        logger.info(f"Document uploaded successfully: {file.filename} (ID: {document_id})")
        
        return StandardResponse(
            message="Document uploaded successfully",
            data={
                "document_id": document_id,
                "filename": file.filename,
                "status": ProcessingStatus.PENDING.value
            }
        )
        
    except DocumentError:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_document_async(document_id: str, processor: DocumentProcessor) -> None:
    """Process document asynchronously."""
    try:
        document = documents_store.get(document_id)
        if not document:
            logger.error(f"Document not found: {document_id}")
            return
        
        # Update status to processing
        document.processing_status = ProcessingStatus.PROCESSING
        documents_store[document_id] = document
        
        # Process document
        logger.info(f"Starting processing for document: {document_id}")
        
        # Extract text and metadata
        text_content, extracted_metadata = await processor.process_file(document.file_path)
        
        # Update metadata with extracted information
        document.metadata.page_count = extracted_metadata.get('page_count')
        document.metadata.author = extracted_metadata.get('author')
        document.metadata.title = extracted_metadata.get('title')
        document.metadata.subject = extracted_metadata.get('subject')
        document.metadata.creation_date = extracted_metadata.get('creation_date')
        document.metadata.modification_date = extracted_metadata.get('modification_date')
        document.metadata.word_count = len(text_content.split()) if text_content else 0
        document.metadata.character_count = len(text_content) if text_content else 0
        
        # Create chunks
        chunks = await processor.create_chunks(text_content, document_id)
        document.chunks_count = len(chunks)
        document.content = text_content
        
        # Update status to completed
        document.processing_status = ProcessingStatus.COMPLETED
        documents_store[document_id] = document
        
        logger.info(f"Document processed successfully: {document_id} ({len(chunks)} chunks)")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        # Update status to failed
        document = documents_store.get(document_id)
        if document:
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = str(e)
            documents_store[document_id] = document


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    status: Optional[ProcessingStatus] = Query(None, description="Filter by processing status"),
    file_type: Optional[DocumentType] = Query(None, description="Filter by file type")
) -> DocumentListResponse:
    """
    List all uploaded documents with pagination and filtering.
    """
    try:
        # Get all documents
        all_documents = list(documents_store.values())
        
        # Apply filters
        filtered_documents = all_documents
        
        if status:
            filtered_documents = [doc for doc in filtered_documents if doc.processing_status == status]
        
        if file_type:
            filtered_documents = [doc for doc in filtered_documents if doc.metadata.file_type == file_type]
        
        # Sort by upload timestamp (newest first)
        filtered_documents.sort(key=lambda x: x.upload_timestamp, reverse=True)
        
        # Calculate pagination
        total_count = len(filtered_documents)
        total_pages = (total_count + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get documents for current page
        page_documents = filtered_documents[start_idx:end_idx]
        
        # Convert to list items
        document_items = [
            DocumentListItem(
                id=doc.id,
                filename=doc.filename,
                file_size=doc.metadata.file_size,
                file_type=doc.metadata.file_type,
                processing_status=doc.processing_status,
                upload_timestamp=doc.upload_timestamp,
                chunks_count=doc.chunks_count,
                author=doc.metadata.author,
                title=doc.metadata.title
            )
            for doc in page_documents
        ]
        
        return DocumentListResponse(
            documents=document_items,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str) -> DocumentStatusResponse:
    """
    Get processing status and details for a specific document.
    """
    try:
        document = documents_store.get(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Calculate progress percentage
        progress_percentage = None
        if document.processing_status == ProcessingStatus.PENDING:
            progress_percentage = 0.0
        elif document.processing_status == ProcessingStatus.PROCESSING:
            progress_percentage = 50.0  # Simplified progress calculation
        elif document.processing_status == ProcessingStatus.COMPLETED:
            progress_percentage = 100.0
        elif document.processing_status == ProcessingStatus.FAILED:
            progress_percentage = 0.0
        
        return DocumentStatusResponse(
            document_id=document_id,
            filename=document.filename,
            processing_status=document.processing_status,
            progress_percentage=progress_percentage,
            error_message=document.error_message,
            chunks_processed=document.chunks_count if document.processing_status == ProcessingStatus.COMPLETED else None,
            total_chunks=document.chunks_count,
            processing_time=None  # Could be calculated if we track start/end times
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")


@router.delete("/{document_id}", response_model=StandardResponse)
async def delete_document(document_id: str) -> StandardResponse:
    """
    Delete a document and its associated files.
    """
    try:
        document = documents_store.get(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file from filesystem
        file_path = Path(document.file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
        
        # Remove from storage
        del documents_store[document_id]
        
        logger.info(f"Document deleted successfully: {document_id}")
        
        return StandardResponse(
            message="Document deleted successfully",
            data={"document_id": document_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str) -> Document:
    """
    Get detailed information about a specific document.
    """
    try:
        document = documents_store.get(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.post("/batch-upload", response_model=StandardResponse)
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    processor: DocumentProcessor = Depends(get_document_processor)
) -> StandardResponse:
    """
    Upload multiple documents in a single request.
    """
    try:
        if len(files) > 10:  # Limit batch size
            raise DocumentError("Maximum 10 files allowed per batch upload")
        
        uploaded_documents = []
        failed_uploads = []
        
        for file in files:
            try:
                # Process each file (simplified version of single upload)
                if not file.filename:
                    failed_uploads.append({"filename": "unknown", "error": "No filename provided"})
                    continue
                
                file_type = get_file_type(file.filename)
                allowed_types = {DocumentType.PDF, DocumentType.DOCX, DocumentType.TXT, DocumentType.MARKDOWN}
                if file_type not in allowed_types:
                    failed_uploads.append({
                        "filename": file.filename,
                        "error": f"Unsupported file type: {Path(file.filename).suffix}"
                    })
                    continue
                
                # Generate unique document ID
                document_id = str(uuid.uuid4())
                
                # Create uploads directory if it doesn't exist
                uploads_dir = Path("uploads")
                uploads_dir.mkdir(exist_ok=True)
                
                # Save file
                file_path = uploads_dir / f"{document_id}_{file.filename}"
                await save_uploaded_file(file, file_path)
                
                # Get file size
                file_size = file_path.stat().st_size
                
                # Create document record
                metadata = DocumentMetadata(
                    filename=file.filename,
                    file_size=file_size,
                    file_type=file_type,
                    mime_type=file.content_type or "application/octet-stream"
                )
                
                document = Document(
                    id=document_id,
                    filename=file.filename,
                    file_path=str(file_path),
                    metadata=metadata,
                    processing_status=ProcessingStatus.PENDING,
                    upload_timestamp=datetime.utcnow()
                )
                
                documents_store[document_id] = document
                uploaded_documents.append({
                    "document_id": document_id,
                    "filename": file.filename,
                    "status": ProcessingStatus.PENDING.value
                })
                
                # Start background processing
                asyncio.create_task(process_document_async(document_id, processor))
                
            except Exception as e:
                failed_uploads.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        response_data = {
            "uploaded_documents": uploaded_documents,
            "successful_uploads": len(uploaded_documents),
            "failed_uploads": len(failed_uploads),
            "failed_details": failed_uploads if failed_uploads else None
        }
        
        if failed_uploads:
            logger.warning(f"Batch upload completed with {len(failed_uploads)} failures")
        else:
            logger.info(f"Batch upload completed successfully: {len(uploaded_documents)} files")
        
        return StandardResponse(
            message=f"Batch upload completed. {len(uploaded_documents)} successful, {len(failed_uploads)} failed.",
            data=response_data
        )
        
    except DocumentError:
        raise
    except Exception as e:
        logger.error(f"Error in batch upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")
