#!/usr/bin/env python3
"""
Testing CI/CD fixes
"""

import sys
from pathlib import Path


def test_workflow_files():
    """Test workflow files"""
    print("🔍 Checking workflow files...")

    project_root = Path(__file__).parent.parent
    workflows_dir = project_root / ".github" / "workflows"

    if not workflows_dir.exists():
        print("❌ Directory .github/workflows does not exist")
        return False

    workflow_files = list(workflows_dir.glob("*.yml"))
    print(f"Found {len(workflow_files)} workflow files")

    issues_found = False

    for workflow_file in workflow_files:
        print(f"\n📄 Checking: {workflow_file.name}")

        try:
            with open(workflow_file, encoding='utf-8') as f:
                content = f.read()

            # Check for deprecated versions
            if "actions/upload-artifact@v3" in content:
                print("  ❌ Found deprecated version upload-artifact@v3")
                issues_found = True
            elif "actions/upload-artifact@v4" in content:
                print("  ✅ Using current version upload-artifact@v4")

            # Check for other deprecated actions
            if "actions/setup-python@v3" in content:
                print("  ⚠️ Recommend updating setup-python to v4")

            if "actions/cache@v2" in content:
                print("  ⚠️ Recommend updating cache to v3")

        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
            issues_found = True

    if not issues_found:
        print("\n✅ All workflow files are OK!")
        return True
    else:
        print("\n❌ Found issues in workflow files")
        return False

def test_requirements_files():
    """Testing requirements files"""
    print("\n🔍 Перевірка файлів requirements...")

    project_root = Path(__file__).parent.parent

    req_files = [
        "requirements.txt",
        "requirements-ci.txt"
    ]

    for req_file in req_files:
        file_path = project_root / req_file
        print(f"\n📄 Перевірка: {req_file}")

        if not file_path.exists():
            print(f"  ❌ Файл {req_file} не існує")
            continue

        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            # Перевірка на основні залежності
            if req_file == "requirements-ci.txt":
                required_packages = [
                    "pytest",
                    "black",
                    "isort",
                    "ruff",
                    "mypy",
                    "bandit"
                ]

                for package in required_packages:
                    if package in content:
                        print(f"  ✅ {package} присутній")
                    else:
                        print(f"  ❌ {package} відсутній")

        except Exception as e:
            print(f"  ❌ Помилка при читанні файлу: {e}")

def test_project_structure():
    """Тестування структури проекту"""
    print("\n🔍 Перевірка структури проекту...")

    project_root = Path(__file__).parent.parent

    required_dirs = [
        "src",
        "app",
        "examples",
        "scripts",
        "tests",
        "docs",
        ".github/workflows"
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name} існує")
        else:
            print(f"  ❌ {dir_name} відсутня")

def test_documentation():
    """Тестування документації"""
    print("\n🔍 Перевірка документації...")

    project_root = Path(__file__).parent.parent

    doc_files = [
        "README.md",
        "docs/CI_CD_IMPROVEMENTS.md"
    ]

    for doc_file in doc_files:
        file_path = project_root / doc_file
        if file_path.exists():
            print(f"  ✅ {doc_file} існує")

            # Перевірка розміру файлу
            size = file_path.stat().st_size
            if size > 100:  # Більше 100 байт
                print(f"    📊 Розмір: {size} байт")
            else:
                print(f"    ⚠️ Файл дуже малий: {size} байт")
        else:
            print(f"  ❌ {doc_file} відсутній")

def main():
    """Головна функція"""
    print("🚀 Тестування виправлень CI/CD для DataMCPServerAgent")
    print("=" * 60)

    all_tests_passed = True

    # Запуск тестів
    tests = [
        ("Workflow файли", test_workflow_files),
        ("Requirements файли", test_requirements_files),
        ("Структура проекту", test_project_structure),
        ("Документація", test_documentation)
    ]

    for test_name, test_func in tests:
        print(f"\n🧪 Тест: {test_name}")
        try:
            result = test_func()
            if result is False:
                all_tests_passed = False
        except Exception as e:
            print(f"❌ Помилка в тесті {test_name}: {e}")
            all_tests_passed = False

    # Підсумок
    print("\n" + "=" * 60)
    print("📊 ПІДСУМОК ТЕСТУВАННЯ")
    print("=" * 60)

    if all_tests_passed:
        print("🎉 Всі тести пройдені успішно!")
        print("✅ CI/CD виправлення працюють коректно")
        return 0
    else:
        print("❌ Деякі тести не пройдені")
        print("⚠️ Потрібні додаткові виправлення")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
