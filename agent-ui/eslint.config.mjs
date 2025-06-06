import { dirname } from 'path'
import { fileURLToPath } from 'url'
import { FlatCompat } from '@eslint/eslintrc'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const compat = new FlatCompat({
  baseDirectory: __dirname
})

const eslintConfig = [
  ...compat.extends('next/core-web-vitals', 'next/typescript'),
  {
    rules: {
      // Disable alt-text rule for Lucide React icons (they are SVG components, not img elements)
      'jsx-a11y/alt-text': 'off',
      // Allow any types in development - can be tightened later
      '@typescript-eslint/no-explicit-any': 'warn'
    }
  }
]

export default eslintConfig
