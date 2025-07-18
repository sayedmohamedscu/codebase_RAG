import ast
import os
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    content: str
    chunk_type: str  # 'function', 'class', 'method', 'import', 'variable', 'comment', 'module'
    name: str
    file_path: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    parent_name: Optional[str] = None
    docstring: Optional[str] = None
    parameters: Optional[List[str]] = None
    return_type: Optional[str] = None
    decorators: Optional[List[str]] = None
    complexity_score: Optional[int] = None
    dependencies: Optional[List[str]] = None
    chunk_id: Optional[str] = None
    
    def __post_init__(self):
        # Generate unique ID based on content and location
        content_hash = hashlib.md5(
            f"{self.file_path}:{self.start_line}:{self.end_line}:{self.content}".encode()
        ).hexdigest()[:8]
        self.chunk_id = f"{self.chunk_type}_{self.name}_{content_hash}"

class CodeChunker:
    """Main class for chunking code using AST"""
    
    def __init__(self, supported_extensions: List[str] = None):
        self.supported_extensions = supported_extensions or ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h']
        self.chunks: List[CodeChunk] = []
        
    def chunk_file(self, file_path: str) -> List[CodeChunk]:
        """Chunk a single file and return list of CodeChunk objects"""
        file_path = Path(file_path)
        
        if file_path.suffix not in self.supported_extensions:
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
        
        if file_path.suffix == '.py':
            return self._chunk_python_file(str(file_path), content)
        else:
            # For other languages, use simpler text-based chunking for now
            return self._chunk_generic_file(str(file_path), content)
    
    def _chunk_python_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Chunk Python file using AST"""
        chunks = []
        lines = content.split('\n')
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return []
        
        # Track imports at module level
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(self._extract_imports(node))
        
        # Process top-level nodes
        for node in tree.body:
            chunk = self._process_node(node, file_path, lines, imports)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _process_node(self, node: ast.AST, file_path: str, lines: List[str], 
                     imports: List[str], parent_name: str = None) -> Optional[CodeChunk]:
        """Process an AST node and create a CodeChunk"""
        
        if isinstance(node, ast.FunctionDef):
            return self._create_function_chunk(node, file_path, lines, imports, parent_name)
        
        elif isinstance(node, ast.AsyncFunctionDef):
            return self._create_function_chunk(node, file_path, lines, imports, parent_name, is_async=True)
        
        elif isinstance(node, ast.ClassDef):
            return self._create_class_chunk(node, file_path, lines, imports)
        
        elif isinstance(node, ast.Assign):
            return self._create_variable_chunk(node, file_path, lines, parent_name)
        
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            return self._create_import_chunk(node, file_path, lines)
        
        return None
    
    def _create_function_chunk(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                              file_path: str, lines: List[str], imports: List[str], 
                              parent_name: str = None, is_async: bool = False) -> CodeChunk:
        """Create a chunk for a function or method"""
        
        # Extract function content
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        content = '\n'.join(lines[start_line-1:end_line])
        
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param_str = arg.arg
            if arg.annotation:
                param_str += f": {ast.unparse(arg.annotation)}"
            parameters.append(param_str)
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            decorators.append(ast.unparse(decorator))
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Calculate complexity (simple metric based on control flow)
        complexity = self._calculate_complexity(node)
        
        chunk_type = "method" if parent_name else "function"
        if is_async:
            chunk_type = "async_" + chunk_type
        
        return CodeChunk(
            content=content,
            chunk_type=chunk_type,
            name=node.name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            start_col=node.col_offset,
            end_col=node.end_col_offset or 0,
            parent_name=parent_name,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            complexity_score=complexity,
            dependencies=imports
        )
    
    def _create_class_chunk(self, node: ast.ClassDef, file_path: str, 
                           lines: List[str], imports: List[str]) -> CodeChunk:
        """Create a chunk for a class"""
        
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        content = '\n'.join(lines[start_line-1:end_line])
        
        # Extract base classes
        base_classes = []
        for base in node.bases:
            base_classes.append(ast.unparse(base))
        
        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            decorators.append(ast.unparse(decorator))
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        return CodeChunk(
            content=content,
            chunk_type="class",
            name=node.name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            start_col=node.col_offset,
            end_col=node.end_col_offset or 0,
            docstring=docstring,
            decorators=decorators,
            dependencies=imports + base_classes
        )
    
    def _create_variable_chunk(self, node: ast.Assign, file_path: str, 
                              lines: List[str], parent_name: str = None) -> Optional[CodeChunk]:
        """Create a chunk for variable assignments"""
        
        # Only process simple assignments at module level
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            content = '\n'.join(lines[start_line-1:end_line])
            
            return CodeChunk(
                content=content,
                chunk_type="variable",
                name=var_name,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                start_col=node.col_offset,
                end_col=node.end_col_offset or 0,
                parent_name=parent_name
            )
        
        return None
    
    def _create_import_chunk(self, node: Union[ast.Import, ast.ImportFrom], 
                            file_path: str, lines: List[str]) -> CodeChunk:
        """Create a chunk for import statements"""
        
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        content = '\n'.join(lines[start_line-1:end_line])
        
        # Extract imported names
        imported_names = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.append(alias.name)
        else:  # ImportFrom
            for alias in node.names:
                imported_names.append(alias.name)
        
        return CodeChunk(
            content=content,
            chunk_type="import",
            name=", ".join(imported_names),
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            start_col=node.col_offset,
            end_col=node.end_col_offset or 0
        )
    
    def _extract_imports(self, node: Union[ast.Import, ast.ImportFrom]) -> List[str]:
        """Extract import names from import nodes"""
        imports = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        else:  # ImportFrom
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1
        
        return complexity
    
    def _chunk_generic_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Generic chunking for non-Python files"""
        chunks = []
        lines = content.split('\n')
        
        # Simple function detection for C/C++/Java/JavaScript
        function_patterns = {
            '.js': r'function\s+(\w+)',
            '.ts': r'function\s+(\w+)',
            '.java': r'(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\(',
            '.cpp': r'\w+\s+(\w+)\s*\(',
            '.c': r'\w+\s+(\w+)\s*\(',
            '.h': r'\w+\s+(\w+)\s*\('
        }
        
        # This is a simplified implementation - you'd want more sophisticated parsing
        # for production use
        
        return chunks
    
    def chunk_directory(self, directory_path: str, recursive: bool = True) -> List[CodeChunk]:
        """Chunk all supported files in a directory"""
        all_chunks = []
        directory_path = Path(directory_path)
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                chunks = self.chunk_file(str(file_path))
                all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        return all_chunks
    
    def save_chunks(self, output_file: str):
        """Save chunks to JSON file"""
        chunks_data = [asdict(chunk) for chunk in self.chunks]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    def load_chunks(self, input_file: str) -> List[CodeChunk]:
        """Load chunks from JSON file"""
        with open(input_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self.chunks = [CodeChunk(**chunk_data) for chunk_data in chunks_data]
        return self.chunks
    
    def get_chunks_by_type(self, chunk_type: str) -> List[CodeChunk]:
        """Filter chunks by type"""
        return [chunk for chunk in self.chunks if chunk.chunk_type == chunk_type]
    
    def get_chunks_by_file(self, file_path: str) -> List[CodeChunk]:
        """Filter chunks by file path"""
        return [chunk for chunk in self.chunks if chunk.file_path == file_path]
    
    def search_chunks(self, query: str) -> List[CodeChunk]:
        """Simple text search in chunks"""
        results = []
        query_lower = query.lower()
        
        for chunk in self.chunks:
            if (query_lower in chunk.content.lower() or 
                query_lower in chunk.name.lower() or
                (chunk.docstring and query_lower in chunk.docstring.lower())):
                results.append(chunk)
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize chunker
    chunker = CodeChunker()
    
    # Example: Chunk a single Python file
    # chunks = chunker.chunk_file("example.py")
    
    # Example: Chunk entire directory
    chunks = chunker.chunk_directory("ultralytics", recursive=True)
    
    # Example: Save chunks to file
    chunker.save_chunks("code_chunks.json")
    
    # Example: Search chunks
    # results = chunker.search_chunks("database")
    
    # Example: Get all functions
    # functions = chunker.get_chunks_by_type("function")
    
    print("Code chunking system initialized!")
    print("Supported file extensions:", chunker.supported_extensions)
    print("\nExample usage:")
    print("1. chunker.chunk_file('path/to/file.py')")
    print("2. chunker.chunk_directory('path/to/project', recursive=True)")
    print("3. chunker.save_chunks('output.json')")
    print("4. chunker.search_chunks('query')")