#!/usr/bin/env python3
"""
Quick fix for basic code issues
"""

import re
from pathlib import Path


def fix_trailing_whitespace(file_path: Path) -> bool:
    """Remove trailing whitespace from lines"""
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        # Remove trailing whitespace
        lines = content.splitlines()
        cleaned_lines = [line.rstrip() for line in lines]
        cleaned_content = '\n'.join(cleaned_lines)

        # Add newline at end of file if missing
        if cleaned_content and not cleaned_content.endswith('\n'):
            cleaned_content += '\n'

        if content != cleaned_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            return True

        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def fix_long_lines(file_path: Path) -> bool:
    """Basic fix for long lines"""
    try:
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        new_lines = []

        for line in lines:
            # Remove trailing whitespace
            clean_line = line.rstrip() + '\n' if line.strip() else '\n'

            # Basic splitting of long lines with comments
            if len(clean_line) > 100 and clean_line.strip().startswith('#'):
                # Splitting long comments
                words = clean_line.strip().split()
                if len(words) > 1:
                    current_line = words[0]
                    for word in words[1:]:
                        if len(current_line + ' ' + word) <= 79:
                            current_line += ' ' + word
                        else:
                            new_lines.append(current_line + '\n')
                            current_line = '# ' + word
                            modified = True
                    new_lines.append(current_line + '\n')
                else:
                    new_lines.append(clean_line)
            else:
                new_lines.append(clean_line)

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

        return modified
    except Exception as e:
        print(f"Error processing long lines in {file_path}: {e}")
        return False

def fix_blank_lines(file_path: Path) -> bool:
    """Fix blank lines with spaces"""
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        # Replace blank lines with spaces with truly blank lines
        fixed_content = re.sub(r'^\s+$', '', content, flags=re.MULTILINE)

        if content != fixed_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True

        return False
    except Exception as e:
        print(f"Error fixing blank lines in {file_path}: {e}")
        return False

def fix_unused_imports(file_path: Path) -> bool:
    """Basic removal of obviously unused imports"""
    try:
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        new_lines = []

        for line in lines:
            # Remove obviously unused imports
            if (line.strip().startswith('from typing import') and
                ('Union' in line or 'List' in line or 'Type' in line)):
                # Check if these types are used in the file
                content = ''.join(lines)

                # Simple usage search
                if 'Union' in line and 'Union[' not in content:
                    line = line.replace('Union, ', '').replace(', Union', '').replace('Union', '')
                    modified = True

                if 'List' in line and 'List[' not in content:
                    line = line.replace('List, ', '').replace(', List', '').replace('List', '')
                    modified = True

                if 'Type' in line and 'Type[' not in content:
                    line = line.replace('Type, ', '').replace(', Type', '').replace('Type', '')
                    modified = True

                # Clean up empty imports
                if line.strip() in ['from typing import', 'from typing import ']:
                    continue

            new_lines.append(line)

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

        return modified
    except Exception as e:
        print(f"Error removing unused imports in {file_path}: {e}")
        return False

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    directories = ["src", "app", "examples", "scripts", "tests"]

    print("🚀 Quick code fixes...")

    total_files = 0
    fixed_files = 0

    for directory in directories:
        dir_path = project_root / directory
        if not dir_path.exists():
            print(f"⚠️ Directory {directory} does not exist")
            continue

        print(f"\n📁 Processing directory: {directory}")

        for py_file in dir_path.rglob("*.py"):
            total_files += 1
            file_fixed = False

            # Fix trailing whitespace
            if fix_trailing_whitespace(py_file):
                file_fixed = True

            # Fix blank lines with spaces
            if fix_blank_lines(py_file):
                file_fixed = True

            # Basic fix for long lines
            if fix_long_lines(py_file):
                file_fixed = True

            # Remove unused imports
            if fix_unused_imports(py_file):
                file_fixed = True

            if file_fixed:
                fixed_files += 1
                print(f"  ✅ Fixed: {py_file.relative_to(project_root)}")

    print("\n📊 Summary:")
    print(f"  Total files: {total_files}")
    print(f"  Fixed files: {fixed_files}")

    if fixed_files > 0:
        print("✅ Fixing completed!")
    else:
        print("ℹ️ No issues found")

if __name__ == "__main__":
    main()
