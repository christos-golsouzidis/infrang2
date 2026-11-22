from pathlib import Path
import shutil
import logging
import io
import pdfminer.high_level as pdfminer
import trafilatura
import openpyxl
import pptx
import docx
import blake3
from qdrant_client import models, QdrantClient
from groq import Groq
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests
import re
import sqlite3
from typing import Literal, Iterable
from tokenizers import Tokenizer
from fastembed import TextEmbedding, SparseTextEmbedding



type FileFormat = Literal['pdf', 'html', 'docx', 'xlsx', 'pptx', 'md', 'txt']

# Custom exceptions for the Infrang class
class InfrangError(Exception):
    """Base exception class for all Infrang-related errors."""
    pass

class InfrangInitializationError(InfrangError):
    """Raised when there's an error initializing the Infrang instance."""
    pass

class InfrangDatabaseError(InfrangError):
    """Raised when there's an error with database operations."""
    pass

class InfrangCollectionError(InfrangError):
    """Raised when there's an error with collection operations."""
    pass

class InfrangDocumentError(InfrangError):
    """Raised when there's an error in processing documents."""
    pass

class InfrangSearchError(InfrangError):
    """Raised when there's an error during search operations."""
    pass

class InfrangGenerationError(InfrangError):
    """Raised when there's an error during answer generation."""
    pass

class InfrangValidationError(InfrangError):
    """Raised when there's a validation error in input parameters."""
    pass


class Infrang:

    def __init__(self,
                collection: str,
                dense_model='BAAI/bge-small-en-v1.5',
                sparse_model='Qdrant/bm42-all-minilm-l6-v2-attentions',
                generative_model='openai/gpt-oss-120b',
                groq_api_key: str | None = None,
                parallel=4,
                infrang_data_dir='.infrang_data',
                max_payload_size=8*1024*1024, # 8MB maximum size
                max_token_chunk=504,
                overlap_token_chunk=16,
                ):
        """Initialize the Infrang instance with the specified document path and model configurations.

            Params:
                collection (str): Name of the collection.
                sparse_model (str): Name of the sparse embedding model.
                dense_model (str): Name of the dense embedding model.
                generative_model (str): Name of the generating model that is used by Groq.
                parallel (int): Number of parallel processes for database operations. Default value is 4.
                infrang_data_dir (str): Directory to store the Qdrant database and SQLite database. Default value is '.infrang_data'.
                max_payload_size (int): Maximum size of the file payload in bytes. Default value is 8MB.
        """
        
        def setup_http():
            """Set up an HTTP session with retry logic for handling transient errors.

            Returns:
                A requests.Session object configured with retry logic.
            """

            self.DEFAULT_HEADERS = {
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:152.0) Gecko/20100101 Firefox/152.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Sec-GPC': '1',
            }
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 501, 502, 503])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            return session

        def setup_init():
            """Set up the Qdrant client."""

            try:
                return QdrantClient(path=str(self.INFRANG_DATA_DIR))
            except Exception as e:
                raise InfrangInitializationError(f'Couldn\'t initialize QDrantClient.') from e

        def test_sqlite_connection() -> Literal[True]:
            """Test the connection to the sqlite database and return true or raise an exception."""

            try:
                conn = sqlite3.connect(self.SOURCES_DB)
                conn.close()
                return True
            except Exception as e:
                raise InfrangInitializationError('Couldn\'t initialize SQLite.') from e
                
        def preload_models() -> None:
            """Preload the models."""

            self.logger.debug('Preloading models...')
            models_dir = str(self.MODEL_DIR)
            try:
                self._dense = TextEmbedding(dense_model, cache_dir=models_dir)
                self._sparse = SparseTextEmbedding(sparse_model, cache_dir=models_dir)
                self._tokenizer = Tokenizer.from_pretrained(dense_model)
                self._tokenizer.no_truncation()

                list(self._dense.embed(['warmup']))
                list(self._sparse.embed(['warmup']))
                self._tokenizer.encode('warmup')

                self.logger.debug('Models preloaded successfully.')
            except Exception as e:
                raise InfrangInitializationError(f'Couldn\'t preload models.') from e

        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.collection = self.validate_string(collection)
        self.INFRANG_DATA_DIR = Path(self.validate_string(infrang_data_dir))
        self.DESTINATION_PATH = self.INFRANG_DATA_DIR / 'documents'
        (self.DESTINATION_PATH / self.collection).mkdir(exist_ok=True, parents=True)
        self.SOURCES_DB = self.DESTINATION_PATH / self.collection / 'sources.sqlite'
        self.MAX_PAYLOAD_SIZE = max_payload_size
        self.DEFAULT_HEADERS = None
        self.TIMEOUT = (3.05, 10)
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        if not isinstance(parallel, int) or parallel < 1:
            raise InfrangValidationError("Parallel must be a positive integer.")
        # do not expose this attribute!
        self._groq = None
        if groq_api_key:
            self.set_groq_client(groq_api_key)
        else:
            self.logger.warning('Groq API key not provided. Set it up when calling \'set_groq_client(self, api_key: str)\'')
        self.parallel = parallel
        self.MAX_TOKEN_CHUNK = max_token_chunk
        self.OVERLAP_TOKEN_CHUNK = overlap_token_chunk
        if self.OVERLAP_TOKEN_CHUNK >= self.MAX_TOKEN_CHUNK:
            raise ValueError('Overlap size must be smaller than chunking size.')
        self.qdrant_client = setup_init()
        test_sqlite_connection()
        self.generative_model = generative_model
        self.session = setup_http()
        self.MODEL_DIR = self.INFRANG_DATA_DIR / 'models'
        self._dense = None
        self._sparse = None
        self._tokenizer: Tokenizer | None = None
        self._SUPPORTED_FORMATS = {'pdf', 'html', 'docx', 'xlsx', 'pptx', 'md', 'txt'}
        preload_models()


    def uninit(self):
        """Uninitializes the Infrang instance and closes the connection to QDrant DB."""

        try:
            self.qdrant_client.close()
        except Exception as e:
            raise InfrangError(f'Error closing QDrantClient: {e}')
        try:
            if self._groq: self._groq.close()
        except Exception as e:
            raise InfrangError(f'Error closing Groq client: {e}')
        try:
            self.session.close()
        except Exception as e:
            raise InfrangError(f'Error closing HTTP session: {e}')


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.uninit()
        except InfrangDatabaseError as e:
            self.logger.error(f"Cleanup failed: {e}")
        return False


    def validate_string(self, input_string: str) -> str:
        """Sanitize the input_string for security reasons.
        
        Args:
            input_string (str): The string that will undergo sanitization.

        Returns:
            A string containing only alphanumerical characters and the symbols '_' and '-'.
        """

        if not isinstance(input_string, str) or not input_string.strip():
            raise InfrangValidationError("Name must be a non-empty input_string.")
        # prevent path traversal
        if not re.fullmatch(r'^\.?[A-Za-z0-9_\-]+$', input_string):
            raise InfrangValidationError("Name must contain no special characters except underscores and hyphens.")
        return input_string


    def get_collections(self):
        """Returns a list of existing collections in the Qdrant database."""

        try:
            return [dir.name for dir in self.DESTINATION_PATH.iterdir()]
        except StopIteration:
            return [] # Return an empty list if no directories exist


    def fetch_url(self, url: str)-> tuple[bytes, str, FileFormat]:
        """Fetch a URL and return its raw body.

        Args:
            url: The URL to fetch.
            headers: Optional headers merged over the defaults.

        Returns:
            A tuple of the HTML Document (bytes), the final URL after redirects (str) and the file format (FileFormat).
        """

        try:
            response = self.session.get(url, headers=self.DEFAULT_HEADERS, timeout=self.TIMEOUT)
            response.raise_for_status()
        except requests.Timeout as e:
            raise InfrangInitializationError(f"Timed out fetching {url}") from e
        except requests.HTTPError as e:
            raise InfrangInitializationError(f"HTTP {e.response.status_code} for {url}") from e # type: ignore
        except requests.RequestException as e:
            raise InfrangInitializationError(f"Request failed for {url}: {e}") from e

        content_type = response.headers.get('Content-Type', '')
        if 'html' in content_type.lower():
            return response.content, response.url, 'html'
        elif 'pdf' in content_type.lower():
            return response.content, response.url, 'pdf'
        else:
            raise InfrangDocumentError('Unsupported type of document.')


    def get_sources(self) -> list[dict]:
        """Get all sources from the current collection."""

        try:
            sources = self._sqlite_get_sources()
            return [{
                'hash': source[0],
                'filename' : source[1],
                'open_status' : source[2]
                } for source in sources]
        except Exception as e:
            raise InfrangDatabaseError('Error accessing the table \'sources\' in the database. Did you create a collection?') from e


    def read_path(self, path: str):
        """Read all files from the specified path and yield their content, filename, and filetype.
        
        Args:
            path (str): The path to the knowledge base.

        Returns:
            A tuple with the payload (bytes), the filename (str) and the filetype (str).
        """

        def iter_files(files):
            """Read lazily each file."""

            for file in files:
                try:
                    fread = file.read_bytes()
                except OSError as e:
                    raise InfrangDocumentError(f'Cannot access the file {file.name}: {e}') from e
                yield (fread, file.name, file.suffix.lstrip('.'))
            
        directory = Path(path)

        if not directory.exists():
            raise InfrangDocumentError(f'Path does not exist: {path}')
        if not directory.is_dir():
            raise InfrangDocumentError(f'Path is not a directory: {path}')

        files = [f for f in directory.glob('*') if f.is_file()]
        if not files:
            raise InfrangDocumentError(f'No files found in: {path}')

        return iter_files(files)


    def get_chunk_number(self, reference_id: str | None=None, source: str | None=None):
        """Get the number of chunks of all sources in the collection.
        
        Args:
            source (str): The document for the chunk number to get. 
                if \'None\' is used (default value), the chunk number of all documents in the database will be returned.

        Returns:
            The number of chunks.
        """
        
        try:
            if not reference_id and not source:
                return self.qdrant_client.count(self.collection).count
            elif reference_id:
                return self.qdrant_client.count(self.collection,
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="reference_id",
                            match=models.MatchValue(value=reference_id)
                            ),
                        ],
                    )
                ).count
            elif source:
                return self.qdrant_client.count(self.collection,
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source)
                            ),
                        ],
                    )
                ).count
        except Exception as e:
            raise InfrangCollectionError(f'Collection \'{self.collection}\' doesn\'t exist.') from e
        

    def _sqlite_get_open_status(self, doc_hash: str):
        """Get the open status of the document in the sources database with the specified hash."""

        conn = sqlite3.connect(self.SOURCES_DB)
        try:
            row = conn.execute(
                'SELECT open_status FROM sources WHERE hash = ?', (doc_hash,)
            ).fetchone()
            return row[0] if row is not None else None
        except Exception as e:
            raise InfrangDatabaseError(f'Error while trying to read from sources. Did you create the collection?') from e
            
        finally:
            conn.close()


    def _sqlite_get_sources(self):
        """Get all the sources from the sources database."""

        conn = sqlite3.connect(self.SOURCES_DB)
        try:
            return conn.execute(
                'SELECT hash, filename, open_status FROM sources'
            ).fetchall()
        finally:
            conn.close()


    def _sqlite_insert_source(self, doc_hash, filename) -> bool:
        """Insert a source. Return False if the hash already exists."""

        conn = sqlite3.connect(self.SOURCES_DB)
        try:
            cursor = conn.execute(
                'INSERT INTO sources (hash, filename, open_status) '
                'VALUES (?, ?, ?) ON CONFLICT(hash) DO NOTHING;',
                (doc_hash, filename, True),
            )
            conn.commit()
            return cursor.rowcount == 1
        finally:
            conn.close()


    def _sqlite_update_open_status(self, doc_hash):
        """Set the open status of a source to 'False'."""

        conn = sqlite3.connect(self.SOURCES_DB)
        try:
            conn.execute(
                'UPDATE sources SET open_status = ? WHERE hash = ?;', (False, doc_hash)
            )
            conn.commit()
        finally:
            conn.close()

    
    def _sqlite_create_table(self):
        """Create the sources table if it doesn't exist."""

        conn = sqlite3.connect(self.SOURCES_DB)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sources (
                    hash TEXT PRIMARY KEY,
                    filename TEXT,
                    open_status BOOL
                )
            ''')
            conn.commit()
        finally:
            conn.close()


    def _assign_pk(self, payload: bytes | str) -> str:
        """Hash the payload and return a string in hexadecimal format."""

        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        return blake3.blake3(payload).hexdigest()


    def _chunk_text(self, text: str, source: str, doc_hash: str) -> Iterable[dict[str, str]]:
        """Chunk the text into smaller pieces for processing.
        
        Args:
            text (str): The text to be chunked.
            source (str): The filename or url to identify the source.
            doc_hash (str): The payload hash which serves as a reference id.

        Returns:
            An iterable of dictionaries, each containing a chunk of text, its source and the reference id to the document.
        """

        if not self._tokenizer:
            raise Exception('Tokenizer is not set.')
        offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        n = len(offsets)
        step = self.MAX_TOKEN_CHUNK - self.OVERLAP_TOKEN_CHUNK
        chunks = []
        for start in range(0, n, step):
            end = min(start + self.MAX_TOKEN_CHUNK, n)
            chunks.append({
                'text': text[offsets[start][0]:offsets[end - 1][1]],
                'source': source,
                'reference_id': doc_hash,
            })
            if end == n:
                break
        return chunks


    def _convert_text(self, payload: bytes, doc_type: FileFormat) -> str:
        """Convert the payload to text based on the document type.
        
        Args:
            payload (bytes): The payload in bytes which is the content for processing.
            doc_type (FileFormat): The type of the source document.

        Returns:
            The extracted text as a string.

        Raises:
            InfrangDocumentError: If the document type is unsupported.
        """


        try:
            if doc_type.lower() not in self._SUPPORTED_FORMATS:
                raise InfrangDocumentError(f'Unsupported media type: {doc_type.lower()}')
            
            if doc_type.lower() == 'pdf':
                return pdfminer.extract_text(io.BytesIO(payload)).strip()
            
            elif doc_type.lower() == 'html':
                return (trafilatura.extract(payload) or '').strip()
            
            elif doc_type.lower() == 'docx':
                return '\n'.join([paragraph.text for paragraph in docx.Document(io.BytesIO(payload)).paragraphs]).strip()
            
            elif doc_type.lower() == 'xlsx':
                wb = openpyxl.load_workbook(io.BytesIO(payload), data_only=True, read_only=True)
                try:
                    output: list[str] = []
                    for sheet in wb.worksheets:
                        output.append(f"--- Sheet: {sheet.title} ---")
                        for row in sheet.iter_rows(values_only=True):
                            line = "\t".join(str(c) if c is not None else "" for c in row).rstrip()
                            if line:
                                output.append(line)
                    return "\n".join(output)
                finally:
                    wb.close()

            elif doc_type.lower() == 'pptx':
                prs = pptx.Presentation(io.BytesIO(payload))
                lines: list[str] = []
                for slide in prs.slides:
                    if lines:
                        lines.append("")
                    for shape in slide.shapes:
                        if shape.has_text_frame and shape.text.strip():
                            lines.append(shape.text.strip())
                return "\n".join(lines).strip()
            
            elif doc_type.lower() in ('md', 'txt'):
                return payload.decode('utf-8', errors='strict').strip() # if not UTF-8 raise

        except Exception as e:
            raise InfrangDocumentError(f'The conversion from \'{doc_type.lower()}\' type was unsuccessful: {e}') from e

    
    def _upsert_chunks(self, payload: Iterable[dict[str, str]]) -> None:
        """Upserts the chunks into the Qdrant collection.
        
        Args:
            payload (Iterable[dict[str, str]]): A Iterable of dictionaries containing the text, 
            the source, and the id reference to the document of each chunk.

        raises:
            InfrangDatabaseError if the upload to qdrant was unsuccessful.
        """
        try:
            self.qdrant_client.upload_collection(
                collection_name=self.collection,
                vectors=[
                    {
                        'dense': models.Document(text=doc['text'], model=self.dense_model),
                        'sparse': models.Document(text=doc['text'], model=self.sparse_model)
                    } for doc in payload],
                payload=payload,
                parallel=self.parallel,
            )
        except Exception as e:
            raise InfrangDatabaseError(f'The chunks upload was unsuccessful: {e}') 


    def _process_doc(self, payload: bytes, source: str, doc_type: FileFormat, doc_hash: str) -> None:
        """Process text according to the type of the source.

        Args:
            payload (bytes): The file / stream object that contains the content for processing.
            source (str): The filename or url to identify the source.
            doc_type (str): The type of the source document.
            doc_hash (str): The payload hash which serves as a reference id.

        Raises:
            InfrangDocumentError: If text for processing is empty.
        """

        t = time.perf_counter()
        text = self._convert_text(payload, doc_type)
        self.logger.info(f'Extract: {time.perf_counter() - t}s')
        if not text:
            raise InfrangDocumentError(f'Processed text in {source} is empty.')
        t = time.perf_counter()
        chunks = self._chunk_text(text, source, doc_hash)
        self.logger.info(f'Chunk: {time.perf_counter() - t}s')
        t = time.perf_counter()
        self._upsert_chunks(chunks)
        self.logger.info(f'Upsert: {time.perf_counter() - t}s')

        
    def _skip_reason(self, content) -> str | None:
        """Return why a source can't be ingested, or None if it's fine."""

        if not content:
            return "empty content"
        if len(content) > self.MAX_PAYLOAD_SIZE:
            mb = 1024 * 1024
            return (
                f"size {len(content) / mb:.1f} MB exceeds the maximum "
                f"of {self.MAX_PAYLOAD_SIZE / mb:.1f} MB"
            )
        return None
    

    def _ingest_source(self, payload: bytes, filename: str, doc_type: FileFormat) -> bool:
        """Ingest one source. Return False if it was skipped."""

        doc_hash = self._assign_pk(payload)
        open_status = self._sqlite_get_open_status(doc_hash)  # None = unknown hash

        if open_status is None:
            pass # hash not stored, ingest it
        elif not open_status:
            # this source was already ingested
            self.logger.warning("Skipping %s (%s...): already ingested.", filename, doc_hash[:8])
            return False
        else:
            # an earlier ingest was interrupted: clean up and redo it
            self.logger.warning("Restoring %s (%s...): previous ingest did not finish.", filename, doc_hash[:8])
            self.remove(reference_id=doc_hash)
        try:
            self._sqlite_insert_source(doc_hash=doc_hash, filename=filename)
            self._process_doc(payload, filename, doc_type, doc_hash)
        except Exception as e:
            self.logger.error("Error ingesting %s: %s", filename, e)
            return False
        
        self._sqlite_update_open_status(doc_hash)
        return True


    def update(self, sources: Iterable[tuple[bytes, str, FileFormat]]):
        """Ingest multiple uploads, skipping (and reporting) any that fail.

        Args:
            sources: An iterable of tuples that contains the content of the sources and the filenames.
                A tuple should contain the payload, the filename and the filetype.
        """
        for content, filename, doc_type in sources:
            reason = self._skip_reason(content)
            if reason:
                self.logger.warning("Skipping %s: %s.", filename, reason)
                continue
            self._ingest_source(content, filename, doc_type)


    def create(self, sources: Iterable[tuple[bytes, str, FileFormat]] | None, overwrite=False):
        """Create a new collection of documents and store them in the database.
        
        Args:
            sources (Iterable[dict[str, str]]): An Iterable of tuples containing the payload (bytes),
                the filename or url to identify the source (str) and the type of the source document (FileFormat type)
                or None. If None is given, an empty collection will be created.
            overwrite (bool): Whether an existing collection should be overwritten or not. Default value is False.
        """
        
        if overwrite:
            self.delete()
        if not (self.DESTINATION_PATH / self.collection).exists():
            (self.DESTINATION_PATH / self.collection).mkdir(exist_ok=True)
            self.SOURCES_DB = self.DESTINATION_PATH / self.collection / 'sources.sqlite'
        self._sqlite_create_table()
        self.qdrant_client.create_collection(
            collection_name=self.collection,
            vectors_config={'dense': models.VectorParams(
                    size=self.qdrant_client.get_embedding_size(self.dense_model), 
                    distance=models.Distance.COSINE
                )},
            sparse_vectors_config={'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF)}
            )
        if sources:
            self.update(sources)


    def delete(self):
        """Delete the default collection from the database."""

        try:
            shutil.rmtree(self.DESTINATION_PATH / self.collection)
            self.qdrant_client.delete_collection(collection_name=self.collection)
        except Exception as e:
            raise InfrangDatabaseError(f"Error in database operations: {e}")

        
    def _sqlite_remove_source(self, column: Literal['hash' , 'filename'], value: str):
        conn = sqlite3.connect(self.SOURCES_DB)
        try:
            conn.execute(
                f'DELETE FROM sources WHERE {column} = ?;', (value,)
            )
            conn.commit()
        except Exception as e:
            raise InfrangDatabaseError("Error in database operations") from e
        finally:
            conn.close()

        
    def remove(self, *, reference_id: str | None = None, filename: str | None = None):
        """Remove a document and all its chunks from the database."""

        if reference_id:
            points_selector = models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="reference_id",
                            match=models.MatchValue(value=reference_id),
                            ),
                        ],
                    )
                )
            column = "hash"
            value = reference_id
        elif filename:
            points_selector = models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=filename),
                            ),
                        ],
                    )
                )
            column = "filename"
            value = filename
        else:
            raise ValueError('Either \'reference_id\' or \'filename\' must be passed as parameters.')
        try:
            self.qdrant_client.delete(
                collection_name=self.collection,
                points_selector=points_selector,
            )
        except Exception as e:
            raise InfrangDatabaseError("Failed to delete from vector store") from e
        self._sqlite_remove_source(column=column, value=value)


    def set_groq_client(self, api_key: str):
        """Initialize the Groq client.
        
        Args:
            api_key (str): The Groq API key.
        """
        self._groq = Groq(api_key=api_key)


    def search(self, query: str, limit: int=8):
        """
        Performs a semantic search over the stored documents using dense and sparse models.
        
        Args:
            query (str): The query string to search for in the database.
            limit (int): The maximum number of search results to return. Default value is 8.

        Returns:
            A list of dictionaries containing the metadata and score of the search results.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError('Error: query must be a non-empty string.')

        search_result = self.qdrant_client.query_points(
            collection_name=self.collection,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query, model=self.dense_model),
                    using='dense',
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=self.sparse_model),
                    using='sparse',
                ),
            ],
            query_filter=None,
            limit=limit,
        )
        return [{
            'metadata': result.payload,
            'score': result.score,
        }
        for result in search_result.points]


    def generate(self, query: str, context: list[str], groq_api_key: str | None = None):
        """
        Generates an answer based on the provided context using the Groq model.

        Params:
            query (str): The query string to generate an answer for.
            context (list[str]): A list of context strings to provide to the model.
            groq_api_key (str, optional): The Groq API key to use for generating the answer. Defaults to None.
            
        Returns:
            A dictionary containing the generated answer and usage statistics.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError('Error: query must be a non-empty string.')

        if not self._groq:
            if not groq_api_key:
                raise InfrangGenerationError('Groq client is not set and Groq API key is \'None\'.')
            else:
                self.set_groq_client(api_key=groq_api_key)
                
        if not context or context == []:
            return {
                'answer' : "Empty context found.",
                'usage' : {
                    'completion_time': 0,
                    'prompt_time': 0,
                    'total_time': 0,
                    'completion_tokens': 0,
                    'prompt_tokens': 0,
                    'total_tokens': 0,
                }
            }
        
        if not isinstance(context, list):
            context = [context]
        system_prompt = '''
You are an assistant that answers questions STRICTLY based on the user's contexts.
Keep responses concise (1-2 sentences unless more detail is needed).
CRUCIAL: Do not use external knowledge or guess. If the answer is missing, say: "I don't know the answer."
'''
        user_query = '<QUERY>\n' + query + '\n</QUERY>'
        user_query += ''.join(['\n<CONTEXT>\n' + item + '\n</CONTEXT>' for item in context])

        response = self._groq.chat.completions.create(
            messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_query
                    }
            ],
            model=self.generative_model,
        )
        return {
            'answer' : response.choices[0].message.content,
            'usage' : {
                'completion_time': response.usage.completion_time,
                'prompt_time': response.usage.prompt_time,
                'total_time': response.usage.total_time,

                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'total_tokens': response.usage.total_tokens,
            }
        }


    def answer(self, query: str, groq_api_key: str | None = None, num_of_results=8):
        """
        Performs a semantic search over the stored documents using dense and sparse models, and generates an answer 
        based on the retrieved context. Note: 'answer' is equivalent to invoking 'search()' 'and generate()' sequentially.
            Params:
                **query (str):** The query string to search for in the database.
                **num_of_results (int):** The maximum number of search results to retrieve for generating the answer. Default value is 8.
            Returns:
                A dictionary containing the generated answer and usage statistics.
        """
        
        results = self.search(query, limit=num_of_results)
        text_results = [item['metadata']['text'] for item in results]
        return self.generate(query=query, context=text_results, groq_api_key=groq_api_key)

