# Legacy Files

This folder contains the old implementation files that have been replaced by the unified `AcademicAgent` system.

## What was changed (September 2025)

The original system had separate agents and routers for parsing and summarizing. This created complexity and redundancy. The new unified system combines both functionalities into a single, more efficient agent.

## Legacy Files

### Agents (`legacy/agents/`)
- `notes_parser.py` - Old notes parsing agent (replaced by `app/agents/academic_agent.py`)
- `summarizer.py` - Old summarizing agent (replaced by `app/agents/academic_agent.py`)

### Routers (`legacy/routers/`)
- `notes_router.py` - Old notes parsing router (replaced by `app/routers/academic_router.py`)
- `notes_router_new.py` - Alternate notes parsing router (unused)
- `summarizer_router.py` - Old summarizing router (replaced by `app/routers/academic_router.py`)
- `summarizer_router_clean.py` - Alternate summarizing router (unused)

## New Unified System

The new system provides:

1. **Unified Agent**: `app/agents/academic_agent.py`
   - Combines parsing and summarizing functionality
   - Uses Ollama Mistral 7B for both operations
   - Provides better integration between parsing and summarizing

2. **Simplified Router**: `app/routers/academic_router.py`
   - Clean, simple endpoints: `/parse`, `/parse-file`, `/summarize`, `/summarize-file`
   - Enhanced file processing with OCR support
   - Unified error handling

3. **Improved Functionality**:
   - Parse endpoints now automatically generate summaries
   - Better prompt engineering for Ollama Mistral 7B
   - Maintains backward compatibility

## API Changes

**Old Endpoints** (multiple variations):
- `/notes/parse`, `/notes/parse-file`
- `/summarize/`, `/summarize/file`, `/summarize/bullet-points`, `/summarize/abstract`

**New Endpoints** (simplified):
- `POST /parse` - Parse and summarize text
- `POST /parse-file` - Parse and summarize files
- `POST /summarize` - Summarize text only
- `POST /summarize-file` - Summarize files only

## Restoration

If you need to restore the old system:
1. Move files back from `legacy/` to their original locations
2. Update `main.py` imports back to the old routers
3. Revert the endpoint changes

However, the new unified system provides all the same functionality with better integration and performance.
