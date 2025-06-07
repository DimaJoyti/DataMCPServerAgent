#!/usr/bin/env python3
"""
Quick fix for basic code issues
"""

import os
import re
from pathlib import Path

def fix_trailing_whitespace(file_path: Path) -> bool:
    """Remove trailing whitespace from lines"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
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
    """Базове виправлення довгих рядків"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        new_lines = []

        for line in lines:
            # Видалення trailing whitespace
            clean_line = line.rstrip() + '\n' if line.strip() else '\n'

            # Базове розбиття довгих рядків з коментарями
            if len(clean_line) > 100 and clean_line.strip().startswith('#'):
                # Розбиття довгих коментарів
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
        print(f"Помилка при обробці довгих рядків в {file_path}: {e}")
        return False

def fix_blank_lines(file_path: Path) -> bool:
    """Виправлення пустих рядків з пробілами"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Заміна пустих рядків з пробілами на справді пустіші рядки
        fixed_content = re.sub(r'^\s+$', '', content, flags=re.MULTILINE)

        if content != fixed_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True

        return False
    except Exception as e:
        print(f"Помилка при виправленні пустих рядків в {file_path}: {e}")
        return False

def fix_unused_imports(file_path: Path) -> bool:
    """Базове видалення очевидно невикористаних імпортів"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        new_lines = []

        for line in lines:
            # Видалення очевидно невикористаних імпортів
            if (line.strip().startswith('from typing import') and
                ('Union' in line or 'List' in line or 'Type' in line)):
                # Перевіряємо чи використовуються ці типи в файлі
                content = ''.join(lines)

                # Простий пошук використання
                if 'Union' in line and 'Union[' not in content:
                    line = line.replace('Union, ', '').replace(', Union', '').replace('Union', '')
                    modified = True

                if 'List' in line and 'List[' not in content:
                    line = line.replace('List, ', '').replace(', List', '').replace('List', '')
                    modified = True

                if 'Type' in line and 'Type[' not in content:
                    line = line.replace('Type, ', '').replace(', Type', '').replace('Type', '')
                    modified = True

                # Очищення пустих імпортів
                if line.strip() in ['from typing import', 'from typing import ']:
                    continue

            new_lines.append(line)

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

        return modified
    except Exception as e:
        print(f"Помилка при видаленні невикористаних імпортів в {file_path}: {e}")
        return False

def main():
    """Головна функція"""
    project_root = Path(__file__).parent.parent
    directories = ["src", "app", "examples", "scripts", "tests"]

    print("🚀 Швидке виправлення проблем коду...")

    total_files = 0
    fixed_files = 0

    for directory in directories:
        dir_path = project_root / directory
        if not dir_path.exists():
            print(f"⚠️ Директорія {directory} не існує")
            continue

        print(f"\n📁 Обробка директорії: {directory}")

        for py_file in dir_path.rglob("*.py"):
            total_files += 1
            file_fixed = False

            # Виправлення trailing whitespace
            if fix_trailing_whitespace(py_file):
                file_fixed = True

            # Виправлення пустих рядків з пробілами
            if fix_blank_lines(py_file):
                file_fixed = True

            # Базове виправлення довгих рядків
            if fix_long_lines(py_file):
                file_fixed = True

            # Видалення невикористаних імпортів
            if fix_unused_imports(py_file):
                file_fixed = True

            if file_fixed:
                fixed_files += 1
                print(f"  ✅ Виправлено: {py_file.relative_to(project_root)}")

    print(f"\n📊 Підсумок:")
    print(f"  Всього файлів: {total_files}")
    print(f"  Виправлено файлів: {fixed_files}")

    if fixed_files > 0:
        print("✅ Виправлення завершено!")
    else:
        print("ℹ️ Проблем не знайдено")

if __name__ == "__main__":
    main()
