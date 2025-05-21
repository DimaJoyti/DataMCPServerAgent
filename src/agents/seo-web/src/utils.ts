import type { ToolCall } from "ai";

/**
 * Process tool calls from the AI model
 */
export async function processToolCalls(
  call: ToolCall,
  executions: Record<string, Function>,
  env: any
): Promise<string> {
  const { name, parameters } = call;
  
  // Check if the tool execution function exists
  if (executions[name]) {
    try {
      // Execute the tool with the provided parameters
      return await executions[name](parameters, env);
    } catch (error) {
      console.error(`Error executing tool ${name}:`, error);
      return `Error executing ${name}: ${error instanceof Error ? error.message : String(error)}`;
    }
  }
  
  return `Tool ${name} not implemented`;
}

/**
 * Format a date for display
 */
export function formatDate(date: Date): string {
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

/**
 * Format a number with commas
 */
export function formatNumber(num: number): string {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Calculate the percentage change between two numbers
 */
export function calculatePercentageChange(current: number, previous: number): number {
  if (previous === 0) return current > 0 ? 100 : 0;
  return ((current - previous) / previous) * 100;
}

/**
 * Format a percentage change with a + or - sign
 */
export function formatPercentageChange(change: number): string {
  const sign = change >= 0 ? "+" : "";
  return `${sign}${change.toFixed(2)}%`;
}

/**
 * Truncate a string to a maximum length
 */
export function truncateString(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return str.substring(0, maxLength - 3) + "...";
}

/**
 * Extract domain from a URL
 */
export function extractDomain(url: string): string {
  try {
    const parsedUrl = new URL(url);
    return parsedUrl.hostname;
  } catch (error) {
    // If the URL is invalid, return the original string
    return url;
  }
}

/**
 * Generate a color based on a value (for charts)
 */
export function generateColor(value: number, min: number, max: number): string {
  // Normalize the value between 0 and 1
  const normalized = (value - min) / (max - min);
  
  // Generate a color from red (0) to green (1)
  const r = Math.round(255 * (1 - normalized));
  const g = Math.round(255 * normalized);
  const b = 0;
  
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Generate a unique ID
 */
export function generateUniqueId(): string {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}

/**
 * Parse a comma-separated string into an array
 */
export function parseCommaSeparatedString(str: string): string[] {
  return str.split(",").map(item => item.trim()).filter(Boolean);
}

/**
 * Format bytes to a human-readable string
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return "0 Bytes";
  
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"];
  
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

/**
 * Format milliseconds to a human-readable string
 */
export function formatMilliseconds(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  
  const minutes = Math.floor(ms / 60000);
  const seconds = ((ms % 60000) / 1000).toFixed(2);
  
  return `${minutes}m ${seconds}s`;
}

/**
 * Calculate the average of an array of numbers
 */
export function calculateAverage(numbers: number[]): number {
  if (numbers.length === 0) return 0;
  return numbers.reduce((sum, num) => sum + num, 0) / numbers.length;
}

/**
 * Calculate the median of an array of numbers
 */
export function calculateMedian(numbers: number[]): number {
  if (numbers.length === 0) return 0;
  
  const sorted = [...numbers].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  
  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }
  
  return sorted[middle];
}
