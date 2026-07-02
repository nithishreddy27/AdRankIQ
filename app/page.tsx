"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { TrendingUp, TrendingDown, Target, ShoppingCart, Activity, Zap, Brain, Database, Cpu } from "lucide-react"

// Mock data - in production this would come from your Python backend
const mockModelResults = {
  ctr_xgboost: { auc: 0.9234, logloss: 0.1456, pr_auc: 0.8901 },
  ctr_neural_network: { auc: 0.9187, logloss: 0.1523, pr_auc: 0.8834 },
  cvr_xgboost: { auc: 0.8967, logloss: 0.2134, pr_auc: 0.8456 },
  cvr_neural_network: { auc: 0.8923, logloss: 0.2201, pr_auc: 0.8389 },
  multitask: { ctr_auc: 0.9156, cvr_auc: 0.8945, combined_auc: 0.9051 },
}

const mockRankingResults = [
  {
    rank: 1,
    item_id: "item_1",
    final_score: 0.8945,
    ctr_prediction: 0.234,
    cvr_prediction: 0.089,
    expected_revenue: 12.45,
    category: "electronics",
  },
  {
    rank: 2,
    item_id: "item_2",
    final_score: 0.8723,
    ctr_prediction: 0.198,
    cvr_prediction: 0.095,
    expected_revenue: 15.67,
    category: "fashion",
  },
  {
    rank: 3,
    item_id: "item_3",
    final_score: 0.8567,
    ctr_prediction: 0.245,
    cvr_prediction: 0.078,
    expected_revenue: 9.23,
    category: "home",
  },
  {
    rank: 4,
    item_id: "item_4",
    final_score: 0.8434,
    ctr_prediction: 0.189,
    cvr_prediction: 0.092,
    expected_revenue: 18.9,
    category: "books",
  },
  {
    rank: 5,
    item_id: "item_5",
    final_score: 0.8321,
    ctr_prediction: 0.212,
    cvr_prediction: 0.085,
    expected_revenue: 11.34,
    category: "beauty",
  },
]

const mockTrainingHistory = [
  { epoch: 0, train_loss: 0.693, val_loss: 0.689, val_auc: 0.512 },
  { epoch: 10, train_loss: 0.456, val_loss: 0.478, val_auc: 0.734 },
  { epoch: 20, train_loss: 0.321, val_loss: 0.356, val_auc: 0.823 },
  { epoch: 30, train_loss: 0.234, val_loss: 0.289, val_auc: 0.867 },
  { epoch: 40, train_loss: 0.189, val_loss: 0.245, val_auc: 0.891 },
  { epoch: 50, train_loss: 0.156, val_loss: 0.223, val_auc: 0.908 },
  { epoch: 60, train_loss: 0.134, val_loss: 0.212, val_auc: 0.918 },
  { epoch: 70, train_loss: 0.123, val_loss: 0.208, val_auc: 0.923 },
]

const mockABTestResults = {
  control: { ctr: 0.145, cvr: 0.078, revenue_per_user: 4.23 },
  treatment: { ctr: 0.156, cvr: 0.084, revenue_per_user: 4.67 },
  lift: { ctr: 7.6, cvr: 7.7, revenue_per_user: 10.4 },
}

const categoryColors = {
  electronics: "#4f46e5",
  fashion: "#7c3aed",
  home: "#06b6d4",
  books: "#f59e0b",
  beauty: "#f43f5e",
}

export default function MLDashboard() {
  const [isRunning, setIsRunning] = useState(false)
  const [currentResults, setCurrentResults] = useState(mockModelResults)

  const runModels = async () => {
    setIsRunning(true)

    // Simulate model training/prediction
    await new Promise((resolve) => setTimeout(resolve, 3000))

    // Update results with slight variations
    const newResults = {
      ...mockModelResults,
      ctr_xgboost: {
        ...mockModelResults.ctr_xgboost,
        auc: 0.9234 + (Math.random() - 0.5) * 0.01,
      },
    }

    setCurrentResults(newResults)
    setIsRunning(false)
  }

  const MetricCard = ({ title, value, change, icon: Icon, format = "decimal", accent = "#4f46e5" }) => {
    const formatValue = (val) => {
      if (format === "percentage") return `${(val * 100).toFixed(2)}%`
      if (format === "currency") return `$${val.toFixed(2)}`
      if (format === "number") return val.toLocaleString()
      return val.toFixed(4)
    }

    return (
      <Card className="metric-card">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
          <span
            className="flex h-9 w-9 items-center justify-center rounded-lg"
            style={{ backgroundColor: `${accent}1a`, color: accent }}
          >
            <Icon className="h-4 w-4" />
          </span>
        </CardHeader>
        <CardContent>
          <div className="metric-value">{formatValue(value)}</div>
          {change !== undefined && (
            <p className="text-xs text-muted-foreground flex items-center mt-1">
              {change > 0 ? (
                <TrendingUp className="h-3 w-3 text-primary mr-1" />
              ) : (
                <TrendingDown className="h-3 w-3 text-destructive mr-1" />
              )}
              {Math.abs(change).toFixed(2)}% from last run
            </p>
          )}
        </CardContent>
      </Card>
    )
  }

  const SimpleBarChart = ({ data, title }) => (
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-muted-foreground">{title}</h4>
      {data.map((item, index) => (
        <div key={index} className="flex items-center space-x-3">
          <div className="w-24 text-sm text-muted-foreground">{item.name}</div>
          <div className="flex-1 bg-muted rounded-full h-3">
            <div
              className="bg-primary h-3 rounded-full transition-all duration-500"
              style={{ width: `${(item.auc - 0.8) * 500}%` }}
            />
          </div>
          <div className="w-16 text-sm font-medium">{item.auc.toFixed(4)}</div>
        </div>
      ))}
    </div>
  )

  const SimpleLineChart = ({ data, title }) => (
    <div className="space-y-4">
      <h4 className="text-sm font-medium text-muted-foreground">{title}</h4>
      <div className="grid grid-cols-8 gap-2 h-32 items-end">
        {data.map((item, index) => (
          <div key={index} className="flex flex-col items-center space-y-1">
            <div className="w-full bg-primary rounded-t" style={{ height: `${item.val_auc * 100}%` }} />
            <div className="text-xs text-muted-foreground">{item.epoch}</div>
          </div>
        ))}
      </div>
    </div>
  )

  const SimplePieChart = ({ data, title }) => (
    <div className="space-y-4">
      <h4 className="text-sm font-medium text-muted-foreground">{title}</h4>
      <div className="grid grid-cols-2 gap-4">
        {data.map((item, index) => (
          <div key={index} className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: categoryColors[item.category] }} />
            <span className="text-sm">
              {item.category}: {item.count}
            </span>
          </div>
        ))}
      </div>
    </div>
  )

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="hero-gradient border-b border-border backdrop-blur supports-[backdrop-filter]:bg-card/40">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="flex items-center gap-3">
              <span className="brand-badge">
                <Cpu className="h-6 w-6" />
              </span>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">
                  <span className="brand-gradient">AdRankIQ</span>
                </h1>
                <p className="text-sm text-muted-foreground">CTR / CVR Prediction &amp; Intelligent Ad Ranking</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="status-success">
                <Activity className="h-3 w-3 mr-1" />
                Models Active
              </Badge>
              <Button onClick={runModels} disabled={isRunning} className="bg-primary hover:bg-primary/90">
                {isRunning ? (
                  <>
                    <Zap className="h-4 w-4 mr-2 animate-spin" />
                    Running Models...
                  </>
                ) : (
                  <>
                    <Brain className="h-4 w-4 mr-2" />
                    Run Prediction
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2 gap-1 rounded-xl bg-muted/60 p-1 sm:grid-cols-3 lg:grid-cols-5">
            <TabsTrigger value="overview" className="rounded-lg">Overview</TabsTrigger>
            <TabsTrigger value="models" className="rounded-lg">Model Performance</TabsTrigger>
            <TabsTrigger value="ranking" className="rounded-lg">Ranking Results</TabsTrigger>
            <TabsTrigger value="training" className="rounded-lg">Training History</TabsTrigger>
            <TabsTrigger value="experiments" className="rounded-lg">A/B Tests</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <MetricCard
                title="Best CTR AUC"
                value={Math.max(currentResults.ctr_xgboost.auc, currentResults.ctr_neural_network.auc)}
                change={2.3}
                icon={Target}
                accent="#4f46e5"
              />
              <MetricCard
                title="Best CVR AUC"
                value={Math.max(currentResults.cvr_xgboost.auc, currentResults.cvr_neural_network.auc)}
                change={1.8}
                icon={ShoppingCart}
                accent="#7c3aed"
              />
              <MetricCard title="Dataset Size" value={10000000} icon={Database} format="number" accent="#06b6d4" />
              <MetricCard title="Features Engineered" value={127} change={5.2} icon={Cpu} accent="#f59e0b" />
            </div>

            {/* Model Comparison Chart */}
            <Card className="chart-container">
              <CardHeader>
                <CardTitle>Model Performance Comparison</CardTitle>
                <CardDescription>AUC scores across different models and tasks</CardDescription>
              </CardHeader>
              <CardContent>
                <SimpleBarChart
                  title="Model AUC Comparison"
                  data={[
                    { name: "CTR XGBoost", auc: currentResults.ctr_xgboost.auc, type: "CTR" },
                    { name: "CTR Neural Net", auc: currentResults.ctr_neural_network.auc, type: "CTR" },
                    { name: "CVR XGBoost", auc: currentResults.cvr_xgboost.auc, type: "CVR" },
                    { name: "CVR Neural Net", auc: currentResults.cvr_neural_network.auc, type: "CVR" },
                    { name: "Multi-task", auc: currentResults.multitask.combined_auc, type: "Combined" },
                  ]}
                />
              </CardContent>
            </Card>

            {/* Target Achievement */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Target Achievement</CardTitle>
                  <CardDescription>Progress towards AUC = 0.91 target</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>CTR Models</span>
                      <span>
                        {(
                          Math.max(currentResults.ctr_xgboost.auc, currentResults.ctr_neural_network.auc) * 100
                        ).toFixed(2)}
                        %
                      </span>
                    </div>
                    <Progress
                      value={
                        (Math.max(currentResults.ctr_xgboost.auc, currentResults.ctr_neural_network.auc) / 0.91) * 100
                      }
                      className="h-2"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>CVR Models</span>
                      <span>
                        {(
                          Math.max(currentResults.cvr_xgboost.auc, currentResults.cvr_neural_network.auc) * 100
                        ).toFixed(2)}
                        %
                      </span>
                    </div>
                    <Progress
                      value={
                        (Math.max(currentResults.cvr_xgboost.auc, currentResults.cvr_neural_network.auc) / 0.91) * 100
                      }
                      className="h-2"
                    />
                  </div>
                  {Math.max(currentResults.ctr_xgboost.auc, currentResults.ctr_neural_network.auc) >= 0.91 && (
                    <Badge className="status-success">🎯 Target AUC Achieved!</Badge>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>System Status</CardTitle>
                  <CardDescription>Current system health and performance</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Model Training</span>
                    <Badge className="status-success">Complete</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Feature Engineering</span>
                    <Badge className="status-success">Active</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Ranking System</span>
                    <Badge className="status-success">Operational</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">A/B Testing</span>
                    <Badge className="status-warning">In Progress</Badge>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Model Performance Tab */}
          <TabsContent value="models" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* CTR Models */}
              <Card>
                <CardHeader>
                  <CardTitle>CTR Prediction Models</CardTitle>
                  <CardDescription>Click-through rate prediction performance</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                      <div>
                        <p className="font-medium">XGBoost</p>
                        <p className="text-sm text-muted-foreground">Gradient Boosting</p>
                      </div>
                      <div className="text-right">
                        <p className="font-bold text-primary">AUC: {currentResults.ctr_xgboost.auc.toFixed(4)}</p>
                        <p className="text-sm text-muted-foreground">
                          Log Loss: {currentResults.ctr_xgboost.logloss.toFixed(4)}
                        </p>
                      </div>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                      <div>
                        <p className="font-medium">Neural Network</p>
                        <p className="text-sm text-muted-foreground">Deep Learning</p>
                      </div>
                      <div className="text-right">
                        <p className="font-bold text-secondary">
                          AUC: {currentResults.ctr_neural_network.auc.toFixed(4)}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Log Loss: {currentResults.ctr_neural_network.logloss.toFixed(4)}
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* CVR Models */}
              <Card>
                <CardHeader>
                  <CardTitle>CVR Prediction Models</CardTitle>
                  <CardDescription>Conversion rate prediction performance</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                      <div>
                        <p className="font-medium">XGBoost</p>
                        <p className="text-sm text-muted-foreground">Gradient Boosting</p>
                      </div>
                      <div className="text-right">
                        <p className="font-bold text-primary">AUC: {currentResults.cvr_xgboost.auc.toFixed(4)}</p>
                        <p className="text-sm text-muted-foreground">
                          Log Loss: {currentResults.cvr_xgboost.logloss.toFixed(4)}
                        </p>
                      </div>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                      <div>
                        <p className="font-medium">Neural Network</p>
                        <p className="text-sm text-muted-foreground">Deep Learning</p>
                      </div>
                      <div className="text-right">
                        <p className="font-bold text-secondary">
                          AUC: {currentResults.cvr_neural_network.auc.toFixed(4)}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Log Loss: {currentResults.cvr_neural_network.logloss.toFixed(4)}
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Multi-task Model */}
            <Card>
              <CardHeader>
                <CardTitle>Multi-Task Learning Model</CardTitle>
                <CardDescription>Joint CTR and CVR prediction with shared representations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-muted/50 rounded-lg">
                    <p className="text-2xl font-bold text-primary">{currentResults.multitask.ctr_auc.toFixed(4)}</p>
                    <p className="text-sm text-muted-foreground">CTR AUC</p>
                  </div>
                  <div className="text-center p-4 bg-muted/50 rounded-lg">
                    <p className="text-2xl font-bold text-secondary">{currentResults.multitask.cvr_auc.toFixed(4)}</p>
                    <p className="text-sm text-muted-foreground">CVR AUC</p>
                  </div>
                  <div className="text-center p-4 bg-muted/50 rounded-lg">
                    <p className="text-2xl font-bold text-accent">{currentResults.multitask.combined_auc.toFixed(4)}</p>
                    <p className="text-sm text-muted-foreground">Combined AUC</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Ranking Results Tab */}
          <TabsContent value="ranking" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Top Ranked Items</CardTitle>
                <CardDescription>Items ranked by the integrated CTR/CVR system</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="data-table">
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-muted/50">
                        <tr>
                          <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                            Rank
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                            Item ID
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                            Score
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                            CTR
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                            CVR
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                            Revenue
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                            Category
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {mockRankingResults.map((item) => (
                          <tr key={item.rank} className="hover:bg-muted/30">
                            <td className="px-4 py-3 text-sm font-medium">#{item.rank}</td>
                            <td className="px-4 py-3 text-sm font-mono">{item.item_id}</td>
                            <td className="px-4 py-3 text-sm">
                              <div className="flex items-center">
                                <div className="w-16 bg-muted rounded-full h-2 mr-2">
                                  <div
                                    className="bg-primary h-2 rounded-full"
                                    style={{ width: `${item.final_score * 100}%` }}
                                  ></div>
                                </div>
                                {item.final_score.toFixed(4)}
                              </div>
                            </td>
                            <td className="px-4 py-3 text-sm">{(item.ctr_prediction * 100).toFixed(2)}%</td>
                            <td className="px-4 py-3 text-sm">{(item.cvr_prediction * 100).toFixed(2)}%</td>
                            <td className="px-4 py-3 text-sm font-medium">${item.expected_revenue.toFixed(2)}</td>
                            <td className="px-4 py-3 text-sm">
                              <Badge
                                variant="outline"
                                style={{
                                  borderColor: categoryColors[item.category],
                                  color: categoryColors[item.category],
                                }}
                              >
                                {item.category}
                              </Badge>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Category Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Category Distribution</CardTitle>
                <CardDescription>Distribution of top-ranked items by category</CardDescription>
              </CardHeader>
              <CardContent>
                <SimplePieChart
                  title="Category Breakdown"
                  data={Object.entries(
                    mockRankingResults.reduce((acc, item) => {
                      acc[item.category] = (acc[item.category] || 0) + 1
                      return acc
                    }, {}),
                  ).map(([category, count]) => ({ category, count }))}
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Training History Tab */}
          <TabsContent value="training" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Model Training Progress</CardTitle>
                <CardDescription>Training and validation metrics over epochs</CardDescription>
              </CardHeader>
              <CardContent>
                <SimpleLineChart title="Validation AUC Progress" data={mockTrainingHistory} />
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Training Summary</CardTitle>
                  <CardDescription>Key training statistics</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Total Epochs</span>
                    <span className="font-medium">70</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Best Validation AUC</span>
                    <span className="font-medium text-primary">0.9234</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Final Training Loss</span>
                    <span className="font-medium">0.123</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Training Time</span>
                    <span className="font-medium">2h 34m</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Model Architecture</CardTitle>
                  <CardDescription>Neural network configuration</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Input Features</span>
                    <span className="font-medium">127</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Hidden Layers</span>
                    <span className="font-medium">4 (512, 256, 128, 64)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Dropout Rate</span>
                    <span className="font-medium">0.3</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Optimizer</span>
                    <span className="font-medium">Adam (lr=0.001)</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* A/B Tests Tab */}
          <TabsContent value="experiments" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>A/B Test Results</CardTitle>
                <CardDescription>CVR Focus Test - Control vs Treatment</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center p-4 bg-muted/50 rounded-lg">
                    <h3 className="font-semibold mb-4">Control Group</h3>
                    <div className="space-y-2">
                      <div>
                        <p className="text-2xl font-bold">{(mockABTestResults.control.ctr * 100).toFixed(2)}%</p>
                        <p className="text-sm text-muted-foreground">CTR</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold">{(mockABTestResults.control.cvr * 100).toFixed(2)}%</p>
                        <p className="text-sm text-muted-foreground">CVR</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold">${mockABTestResults.control.revenue_per_user.toFixed(2)}</p>
                        <p className="text-sm text-muted-foreground">Revenue/User</p>
                      </div>
                    </div>
                  </div>

                  <div className="text-center p-4 bg-muted/50 rounded-lg">
                    <h3 className="font-semibold mb-4">Treatment Group</h3>
                    <div className="space-y-2">
                      <div>
                        <p className="text-2xl font-bold text-primary">
                          {(mockABTestResults.treatment.ctr * 100).toFixed(2)}%
                        </p>
                        <p className="text-sm text-muted-foreground">CTR</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-primary">
                          {(mockABTestResults.treatment.cvr * 100).toFixed(2)}%
                        </p>
                        <p className="text-sm text-muted-foreground">CVR</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-primary">
                          ${mockABTestResults.treatment.revenue_per_user.toFixed(2)}
                        </p>
                        <p className="text-sm text-muted-foreground">Revenue/User</p>
                      </div>
                    </div>
                  </div>

                  <div className="text-center p-4 bg-muted/50 rounded-lg">
                    <h3 className="font-semibold mb-4">Lift</h3>
                    <div className="space-y-2">
                      <div>
                        <p className="text-2xl font-bold text-secondary">+{mockABTestResults.lift.ctr.toFixed(1)}%</p>
                        <p className="text-sm text-muted-foreground">CTR Lift</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-secondary">+{mockABTestResults.lift.cvr.toFixed(1)}%</p>
                        <p className="text-sm text-muted-foreground">CVR Lift</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-secondary">
                          +{mockABTestResults.lift.revenue_per_user.toFixed(1)}%
                        </p>
                        <p className="text-sm text-muted-foreground">Revenue Lift</p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Statistical Significance</CardTitle>
                <CardDescription>Test confidence and recommendations</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-primary/10 rounded-lg">
                  <div>
                    <p className="font-semibold text-primary">Test Result: Significant</p>
                    <p className="text-sm text-muted-foreground">95% confidence level achieved</p>
                  </div>
                  <Badge className="status-success">Recommend Treatment</Badge>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold mb-2">Test Details</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Sample Size</span>
                        <span>10,000 users</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Test Duration</span>
                        <span>14 days</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Traffic Split</span>
                        <span>50/50</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2">Recommendations</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Deploy treatment to 100% traffic</li>
                      <li>• Expected revenue increase: +10.4%</li>
                      <li>• Monitor for 2 weeks post-deployment</li>
                      <li>• Consider further CVR optimizations</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
