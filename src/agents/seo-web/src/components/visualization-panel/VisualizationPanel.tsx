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
  PolarRadiusAxis,
  Cell
} from 'recharts';
import { 
  ChartLine, 
  ChartBar, 
  ChartPie, 
  ChartDonut,
  Download,
  Share,
  ArrowsClockwise
} from '@phosphor-icons/react';

// Mock data for different visualization types
const mockVisualizationData = {
  keyword_rankings: {
    title: 'Keyword Rankings Over Time',
    description: 'Track the position changes of your top keywords over the last 30 days',
    data: [
      { date: '2023-01-01', 'seo optimization': 12, 'seo tools': 8, 'keyword research': 15 },
      { date: '2023-01-08', 'seo optimization': 10, 'seo tools': 7, 'keyword research': 14 },
      { date: '2023-01-15', 'seo optimization': 8, 'seo tools': 9, 'keyword research': 12 },
      { date: '2023-01-22', 'seo optimization': 6, 'seo tools': 8, 'keyword research': 10 },
      { date: '2023-01-29', 'seo optimization': 5, 'seo tools': 8, 'keyword research': 9 },
      { date: '2023-02-05', 'seo optimization': 5, 'seo tools': 7, 'keyword research': 8 }
    ],
    type: 'line'
  },
  backlink_profile: {
    title: 'Backlink Profile Analysis',
    description: 'Distribution of backlinks by domain authority and type',
    data: [
      { name: 'DA 90-100', dofollow: 5, nofollow: 2 },
      { name: 'DA 80-89', dofollow: 12, nofollow: 5 },
      { name: 'DA 70-79', dofollow: 25, nofollow: 10 },
      { name: 'DA 60-69', dofollow: 40, nofollow: 18 },
      { name: 'DA 50-59', dofollow: 65, nofollow: 30 },
      { name: 'DA 40-49', dofollow: 85, nofollow: 45 },
      { name: 'DA 30-39', dofollow: 120, nofollow: 60 },
      { name: 'DA 20-29', dofollow: 180, nofollow: 90 },
      { name: 'DA 10-19', dofollow: 250, nofollow: 120 },
      { name: 'DA 0-9', dofollow: 350, nofollow: 180 }
    ],
    type: 'bar'
  },
  seo_score_comparison: {
    title: 'SEO Score Comparison',
    description: 'Compare your SEO score with competitors',
    data: [
      { name: 'Your Site', score: 78 },
      { name: 'Competitor 1', score: 85 },
      { name: 'Competitor 2', score: 72 },
      { name: 'Competitor 3', score: 65 },
      { name: 'Industry Average', score: 70 }
    ],
    type: 'bar'
  },
  content_metrics: {
    title: 'Content Metrics Analysis',
    description: 'Breakdown of content metrics across your website',
    data: [
      { name: 'Word Count', value: 1250 },
      { name: 'Headings', value: 25 },
      { name: 'Images', value: 18 },
      { name: 'Internal Links', value: 32 },
      { name: 'External Links', value: 15 }
    ],
    type: 'pie'
  },
  page_speed: {
    title: 'Page Speed Metrics',
    description: 'Core Web Vitals and other page speed metrics',
    data: [
      { metric: 'FCP', mobile: 2.1, desktop: 1.5, benchmark: 1.8 },
      { metric: 'LCP', mobile: 3.5, desktop: 2.2, benchmark: 2.5 },
      { metric: 'CLS', mobile: 0.12, desktop: 0.05, benchmark: 0.1 },
      { metric: 'TTI', mobile: 5.2, desktop: 3.1, benchmark: 3.8 },
      { metric: 'TBT', mobile: 350, desktop: 180, benchmark: 200 }
    ],
    type: 'radar'
  }
};

// Visualization Panel component
export const VisualizationPanel: React.FC<{ data: any }> = ({ data }) => {
  const [activeTab, setActiveTab] = useState('chart');
  const [visualizationType, setVisualizationType] = useState<string>('keyword_rankings');
  const [chartData, setChartData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load visualization data based on the selected type
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // In a real implementation, this would fetch data from an API
        // For now, we'll just use the mock data
        await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API call
        
        // If data is provided, use it to determine the visualization type
        if (data && data.data_type) {
          setVisualizationType(data.data_type);
        }
        
        setChartData(mockVisualizationData[visualizationType as keyof typeof mockVisualizationData]);
      } catch (err) {
        setError('Failed to load visualization data');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, [visualizationType, data]);

  // Handle visualization type change
  const handleVisualizationTypeChange = (type: string) => {
    setVisualizationType(type);
  };

  // Handle download chart
  const handleDownload = () => {
    // In a real implementation, this would generate and download the chart as an image
    alert('Chart download functionality would be implemented here');
  };

  // Handle share chart
  const handleShare = () => {
    // In a real implementation, this would share the chart
    alert('Chart sharing functionality would be implemented here');
  };

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
        <Button onClick={() => setVisualizationType(visualizationType)}>Retry</Button>
      </div>
    );
  }

  // Render empty state
  if (!chartData) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-4">
        <ChartLine size={48} className="text-neutral-400 mb-4" />
        <h2 className="text-xl font-semibold mb-2">No Visualization Data</h2>
        <p className="text-neutral-500 text-center mb-4">
          Select a visualization type to see SEO metrics and insights.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full">
      <div className="mb-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold">{chartData.title}</h2>
          <p className="text-sm text-neutral-500">{chartData.description}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download size={16} className="mr-1" />
            Download
          </Button>
          <Button variant="outline" size="sm" onClick={handleShare}>
            <Share size={16} className="mr-1" />
            Share
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
        <Card className="p-4 col-span-1 md:col-span-3">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="mb-4">
              <TabsTrigger value="chart">Chart</TabsTrigger>
              <TabsTrigger value="data">Data</TabsTrigger>
              <TabsTrigger value="insights">Insights</TabsTrigger>
            </TabsList>

            <TabsContent value="chart" className="h-[500px]">
              {chartData.type === 'line' && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData.data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis reversed />
                    <Tooltip />
                    <Legend />
                    {Object.keys(chartData.data[0]).filter(key => key !== 'date').map((key, index) => (
                      <Line 
                        key={index} 
                        type="monotone" 
                        dataKey={key} 
                        stroke={`#${Math.floor(Math.random() * 16777215).toString(16)}`} 
                        activeDot={{ r: 8 }} 
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}

              {chartData.type === 'bar' && (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData.data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {Object.keys(chartData.data[0]).filter(key => key !== 'name').map((key, index) => (
                      <Bar 
                        key={index} 
                        dataKey={key} 
                        fill={`#${Math.floor(Math.random() * 16777215).toString(16)}`} 
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              )}

              {chartData.type === 'pie' && (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={chartData.data}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={150}
                      fill="#8884d8"
                      label
                    >
                      {chartData.data.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={`#${Math.floor(Math.random() * 16777215).toString(16)}`} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              )}

              {chartData.type === 'radar' && (
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData.data}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" />
                    <PolarRadiusAxis />
                    <Radar name="Mobile" dataKey="mobile" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                    <Radar name="Desktop" dataKey="desktop" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
                    <Radar name="Benchmark" dataKey="benchmark" stroke="#ffc658" fill="#ffc658" fillOpacity={0.6} />
                    <Legend />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              )}
            </TabsContent>

            <TabsContent value="data">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      {Object.keys(chartData.data[0]).map((key, index) => (
                        <th key={index} className="text-left py-2">{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {chartData.data.map((item: any, index: number) => (
                      <tr key={index} className="border-b last:border-b-0">
                        {Object.keys(item).map((key, keyIndex) => (
                          <td key={keyIndex} className="py-2">{item[key]}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </TabsContent>

            <TabsContent value="insights">
              <div className="space-y-4">
                <p>
                  Based on the data visualization, here are some key insights:
                </p>
                <ul className="list-disc pl-5 space-y-2">
                  <li>Your keyword rankings have improved consistently over the past month.</li>
                  <li>The majority of your backlinks come from domains with DA 10-39.</li>
                  <li>Your SEO score is slightly above the industry average but below your top competitor.</li>
                  <li>Your content has a good balance of text, headings, and links.</li>
                  <li>Mobile page speed metrics need improvement, especially LCP and TTI.</li>
                </ul>
                <p>
                  <strong>Recommendations:</strong>
                </p>
                <ul className="list-disc pl-5 space-y-2">
                  <li>Focus on building more high-quality backlinks from domains with DA 40+.</li>
                  <li>Optimize mobile page loading performance to improve Core Web Vitals.</li>
                  <li>Continue your keyword optimization strategy as it's showing positive results.</li>
                </ul>
              </div>
            </TabsContent>
          </Tabs>
        </Card>

        <Card className="p-4">
          <h3 className="font-medium mb-4">Visualization Types</h3>
          <div className="space-y-2">
            <Button
              variant={visualizationType === 'keyword_rankings' ? 'default' : 'outline'}
              className="w-full justify-start"
              onClick={() => handleVisualizationTypeChange('keyword_rankings')}
            >
              <ChartLine size={16} className="mr-2" />
              Keyword Rankings
            </Button>
            <Button
              variant={visualizationType === 'backlink_profile' ? 'default' : 'outline'}
              className="w-full justify-start"
              onClick={() => handleVisualizationTypeChange('backlink_profile')}
            >
              <ChartBar size={16} className="mr-2" />
              Backlink Profile
            </Button>
            <Button
              variant={visualizationType === 'seo_score_comparison' ? 'default' : 'outline'}
              className="w-full justify-start"
              onClick={() => handleVisualizationTypeChange('seo_score_comparison')}
            >
              <ChartBar size={16} className="mr-2" />
              SEO Score Comparison
            </Button>
            <Button
              variant={visualizationType === 'content_metrics' ? 'default' : 'outline'}
              className="w-full justify-start"
              onClick={() => handleVisualizationTypeChange('content_metrics')}
            >
              <ChartPie size={16} className="mr-2" />
              Content Metrics
            </Button>
            <Button
              variant={visualizationType === 'page_speed' ? 'default' : 'outline'}
              className="w-full justify-start"
              onClick={() => handleVisualizationTypeChange('page_speed')}
            >
              <ChartDonut size={16} className="mr-2" />
              Page Speed
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
};
