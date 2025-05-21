import React, { useState, useEffect } from 'react';
import { Card } from '../card/Card';
import { Button } from '../button/Button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../tabs/Tabs';
import { 
  BarChart, 
  LineChart, 
  PieChart, 
  RadarChart,
  Bar,
  Line,
  Pie,
  Radar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from 'recharts';
import { 
  Globe, 
  MagnifyingGlass, 
  ChartLine, 
  ArrowUp, 
  ArrowDown, 
  Minus,
  Lightning,
  Clock,
  DeviceMobile,
  LinkSimple,
  TextT,
  Image,
  Code
} from '@phosphor-icons/react';

// Mock data for the dashboard
const mockSEOData = {
  overview: {
    seoScore: 78,
    pageSpeed: {
      mobile: 65,
      desktop: 82
    },
    issuesCount: {
      critical: 3,
      warning: 8,
      info: 12
    },
    lastUpdated: new Date().toISOString()
  },
  keywordRankings: [
    { keyword: 'seo optimization', position: 5, change: 2 },
    { keyword: 'seo tools', position: 8, change: -1 },
    { keyword: 'keyword research', position: 12, change: 3 },
    { keyword: 'backlink analysis', position: 15, change: 0 },
    { keyword: 'content optimization', position: 7, change: 5 }
  ],
  contentMetrics: {
    wordCount: 1250,
    readabilityScore: 68,
    keywordDensity: [
      { keyword: 'seo', density: 2.4 },
      { keyword: 'optimization', density: 1.8 },
      { keyword: 'content', density: 1.5 },
      { keyword: 'keywords', density: 1.2 },
      { keyword: 'analysis', density: 0.9 }
    ],
    headingStructure: {
      h1: 1,
      h2: 5,
      h3: 12,
      h4: 8
    }
  },
  technicalSEO: {
    issuesByCategory: [
      { name: 'Meta Tags', value: 2 },
      { name: 'Content', value: 5 },
      { name: 'Links', value: 3 },
      { name: 'Images', value: 4 },
      { name: 'Performance', value: 7 },
      { name: 'Mobile', value: 2 }
    ],
    pageSpeedMetrics: [
      { name: 'FCP', score: 85 },
      { name: 'LCP', score: 72 },
      { name: 'CLS', score: 90 },
      { name: 'TTI', score: 68 },
      { name: 'TBT', score: 75 }
    ]
  },
  backlinks: {
    total: 1250,
    dofollow: 850,
    nofollow: 400,
    domainAuthority: 45,
    topBacklinks: [
      { domain: 'example.com', authority: 78 },
      { domain: 'blog.example.org', authority: 65 },
      { domain: 'news.example.net', authority: 58 },
      { domain: 'forum.example.io', authority: 52 },
      { domain: 'review.example.co', authority: 48 }
    ]
  }
};

// SEO Dashboard component
export const SEODashboard: React.FC<{ domain: string | null }> = ({ domain }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [data, setData] = useState(mockSEOData);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch data for the domain
  useEffect(() => {
    if (!domain) return;

    const fetchData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // In a real implementation, this would fetch data from an API
        // For now, we'll just use the mock data
        await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
        setData(mockSEOData);
      } catch (err) {
        setError('Failed to fetch SEO data');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [domain]);

  // Render loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  // Render error state
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-4">
        <div className="text-red-500 mb-4">Error: {error}</div>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    );
  }

  // Render empty state
  if (!domain) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-4">
        <Globe size={48} className="text-neutral-400 mb-4" />
        <h2 className="text-xl font-semibold mb-2">No Domain Selected</h2>
        <p className="text-neutral-500 text-center mb-4">
          Ask the SEO Agent to analyze a website to see detailed metrics and insights.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-4">
      <div className="mb-4">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <Globe size={20} className="text-blue-500" />
          {domain}
        </h2>
        <p className="text-sm text-neutral-500">
          Last updated: {new Date(data.overview.lastUpdated).toLocaleString()}
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="keywords">Keywords</TabsTrigger>
          <TabsTrigger value="content">Content</TabsTrigger>
          <TabsTrigger value="technical">Technical</TabsTrigger>
          <TabsTrigger value="backlinks">Backlinks</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium">SEO Score</h3>
                <MagnifyingGlass size={20} className="text-blue-500" />
              </div>
              <div className="flex items-center justify-center py-4">
                <div className="relative w-32 h-32">
                  <svg className="w-full h-full" viewBox="0 0 36 36">
                    <path
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none"
                      stroke="#eee"
                      strokeWidth="3"
                    />
                    <path
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none"
                      stroke={data.overview.seoScore > 75 ? "#4ade80" : data.overview.seoScore > 50 ? "#facc15" : "#ef4444"}
                      strokeWidth="3"
                      strokeDasharray={`${data.overview.seoScore}, 100`}
                    />
                    <text x="18" y="20.5" textAnchor="middle" className="text-3xl font-bold">
                      {data.overview.seoScore}
                    </text>
                  </svg>
                </div>
              </div>
              <div className="text-center text-sm text-neutral-500">
                {data.overview.seoScore > 75 ? "Good" : data.overview.seoScore > 50 ? "Needs Improvement" : "Poor"}
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium">Page Speed</h3>
                <Lightning size={20} className="text-blue-500" />
              </div>
              <div className="space-y-2">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm">Mobile</span>
                    <span className="text-sm font-medium">{data.overview.pageSpeed.mobile}/100</span>
                  </div>
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${data.overview.pageSpeed.mobile}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm">Desktop</span>
                    <span className="text-sm font-medium">{data.overview.pageSpeed.desktop}/100</span>
                  </div>
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${data.overview.pageSpeed.desktop}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium">Issues</h3>
                <Code size={20} className="text-blue-500" />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-red-500"></span>
                    Critical
                  </span>
                  <span className="text-sm font-medium">{data.overview.issuesCount.critical}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-yellow-500"></span>
                    Warning
                  </span>
                  <span className="text-sm font-medium">{data.overview.issuesCount.warning}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                    Info
                  </span>
                  <span className="text-sm font-medium">{data.overview.issuesCount.info}</span>
                </div>
              </div>
            </Card>
          </div>

          <Card className="p-4">
            <h3 className="font-medium mb-4">Keyword Rankings</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2">Keyword</th>
                    <th className="text-center py-2">Position</th>
                    <th className="text-center py-2">Change</th>
                  </tr>
                </thead>
                <tbody>
                  {data.keywordRankings.slice(0, 5).map((keyword, index) => (
                    <tr key={index} className="border-b last:border-b-0">
                      <td className="py-2">{keyword.keyword}</td>
                      <td className="text-center py-2">{keyword.position}</td>
                      <td className="text-center py-2">
                        {keyword.change > 0 ? (
                          <span className="text-green-500 flex items-center justify-center gap-1">
                            <ArrowUp size={14} />
                            {keyword.change}
                          </span>
                        ) : keyword.change < 0 ? (
                          <span className="text-red-500 flex items-center justify-center gap-1">
                            <ArrowDown size={14} />
                            {Math.abs(keyword.change)}
                          </span>
                        ) : (
                          <span className="text-neutral-500 flex items-center justify-center gap-1">
                            <Minus size={14} />
                            {keyword.change}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </TabsContent>

        {/* Other tabs would be implemented similarly */}
        <TabsContent value="keywords">
          <Card className="p-4">
            <h3 className="font-medium mb-4">Keyword Rankings</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.keywordRankings}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="keyword" />
                <YAxis reversed />
                <Tooltip />
                <Legend />
                <Bar dataKey="position" fill="#3b82f6" name="Position" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>

        <TabsContent value="content">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="p-4">
              <h3 className="font-medium mb-4">Content Metrics</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm">Word Count</span>
                    <span className="text-sm font-medium">{data.contentMetrics.wordCount}</span>
                  </div>
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${Math.min(100, (data.contentMetrics.wordCount / 1500) * 100)}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm">Readability Score</span>
                    <span className="text-sm font-medium">{data.contentMetrics.readabilityScore}/100</span>
                  </div>
                  <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${data.contentMetrics.readabilityScore}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <h3 className="font-medium mb-4">Keyword Density</h3>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={data.contentMetrics.keywordDensity}
                    dataKey="density"
                    nameKey="keyword"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    label
                  >
                    {data.contentMetrics.keywordDensity.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={`#${Math.floor(Math.random() * 16777215).toString(16)}`} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};
