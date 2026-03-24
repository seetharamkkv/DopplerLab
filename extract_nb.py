import json
import sys
import io

# Set stdout to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_definition(nb_path, search_term):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    found = False
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source_lines = cell.get('source', [])
            source_text = "".join(source_lines)
            
            # Check if any line starts with 'class SearchTerm' or 'def SearchTerm'
            is_def = any(line.strip().startswith(f"class {search_term}") or 
                         line.strip().startswith(f"def {search_term}") 
                         for line in source_lines)
            
            if is_def:
                print(f"--- Found definition in cell {i} ---")
                print(source_text)
                print("------------------------------------")
                found = True
    
    if not found:
        # Fallback to simple containment if no explicit definition found
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source_text = "".join(cell.get('source', []))
                if search_term in source_text:
                    print(f"--- Found containment in cell {i} ---")
                    print(source_text)
                    print("------------------------------------")

if __name__ == "__main__":
    extract_definition(sys.argv[1], sys.argv[2])
