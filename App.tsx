
import React, { useState } from 'react';
import { 
  FolderIcon, 
  FileIcon, 
  TerminalIcon, 
  LayoutIcon, 
  BookOpenIcon,
  ShieldCheckIcon,
  BarChart3Icon,
  AlertTriangleIcon,
  CheckCircle2Icon,
  Loader2Icon
} from 'lucide-react';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('readme');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictionResult, setPredictionResult] = useState<{ risk: string, prob: number } | null>(null);

  const files = [
    { path: 'data/raw/example.csv', type: 'data' },
    { path: 'notebooks/01_EDA.ipynb', type: 'notebook' },
    { path: 'reports/data_profile_report.html', type: 'report' },
    { path: 'reports/shap_summary.png', type: 'image' },
    { path: 'src/data_processing.py', type: 'code' },
    { path: 'src/evaluation.py', type: 'code' },
    { path: 'app/streamlit_app.py', type: 'code' },
    { path: 'Dockerfile', type: 'config' },
  ];

  const handleAnalyze = (e: React.FormEvent) => {
    e.preventDefault();
    setIsAnalyzing(true);
    setPredictionResult(null);

    // Simulate model inference delay
    setTimeout(() => {
      setIsAnalyzing(false);
      // Randomly determine risk for mockup purposes
      const prob = Math.random();
      setPredictionResult({
        risk: prob > 0.5 ? 'High Risk' : 'Low Risk',
        prob: parseFloat(prob.toFixed(2))
      });
    }, 1200);
  };

  return (
    <div className="min-h-screen flex flex-col font-sans text-gray-900 bg-gray-50">
      {/* Header */}
      <header className="bg-indigo-700 text-white p-6 shadow-lg">
        <div className="container mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <ShieldCheckIcon size={32} />
            <h1 className="text-2xl font-bold">Credit Risk Prediction System</h1>
          </div>
          <div className="text-sm opacity-80">Author: Anish Choudhary</div>
        </div>
      </header>

      <div className="flex-1 container mx-auto flex gap-6 p-6 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 flex flex-col gap-2">
          <button 
            onClick={() => setActiveTab('readme')}
            className={`flex items-center gap-3 p-3 rounded-lg transition ${activeTab === 'readme' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
          >
            <BookOpenIcon size={20} /> README
          </button>
          <button 
            onClick={() => setActiveTab('structure')}
            className={`flex items-center gap-3 p-3 rounded-lg transition ${activeTab === 'structure' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
          >
            <FolderIcon size={20} /> Project Structure
          </button>
          <button 
            onClick={() => setActiveTab('demo')}
            className={`flex items-center gap-3 p-3 rounded-lg transition ${activeTab === 'demo' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
          >
            <LayoutIcon size={20} /> Streamlit UI Mockup
          </button>
          <hr className="my-2" />
          <div className="px-3 py-2 text-xs font-bold text-gray-400 uppercase tracking-wider">Source Code & Reports</div>
          {files.map(f => (
            <div key={f.path} className="flex items-center gap-2 px-3 py-1 text-sm text-gray-600 truncate">
              <FileIcon size={14} /> {f.path}
            </div>
          ))}
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 bg-white rounded-xl shadow-sm border border-gray-200 p-8 overflow-y-auto">
          {activeTab === 'readme' && (
            <article className="prose max-w-none">
              <h2 className="text-3xl font-bold mb-4">Project Overview</h2>
              <p className="text-gray-600 mb-6">
                This end-to-end ML system predicts credit default risk using a synthetic dataset of loan applicants. 
                It includes a modular data pipeline, hyperparameter tuning for XGBoost, and a Streamlit frontend.
              </p>
              
              <h3 className="text-xl font-semibold mb-3">Model Performance</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg text-center">
                  <div className="text-sm text-green-600 font-bold">ROC-AUC</div>
                  <div className="text-2xl font-mono">0.892</div>
                </div>
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg text-center">
                  <div className="text-sm text-blue-600 font-bold">Recall (High Risk)</div>
                  <div className="text-2xl font-mono">0.84</div>
                </div>
                <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg text-center">
                  <div className="text-sm text-purple-600 font-bold">F1-Score</div>
                  <div className="text-2xl font-mono">0.78</div>
                </div>
              </div>

              <h3 className="text-xl font-semibold mb-3">Model Interpretation</h3>
              <p className="text-gray-600 mb-4">
                We use <strong>SHAP (SHapley Additive exPlanations)</strong> to explain model decisions. A summary plot is generated in <code>reports/shap_summary.png</code> during training.
              </p>

              <h3 className="text-xl font-semibold mb-3">How to run</h3>
              <div className="bg-gray-900 text-gray-300 p-4 rounded-lg font-mono text-sm leading-relaxed overflow-x-auto">
                # Install dependencies<br/>
                pip install -r requirements.txt<br/><br/>
                # Run Tests<br/>
                python -m pytest<br/><br/>
                # Start Streamlit App<br/>
                streamlit run app/streamlit_app.py
              </div>
            </article>
          )}

          {activeTab === 'structure' && (
            <div>
              <h2 className="text-2xl font-bold mb-6">Repository Blueprint</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-gray-50 rounded-lg border border-dashed border-gray-300 hover:border-indigo-400 transition-colors">
                  <span className="font-mono text-indigo-600 font-bold block mb-1">/src</span>
                  <p className="text-sm text-gray-500">Core logic for feature engineering, data loading, and model evaluation.</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg border border-dashed border-gray-300 hover:border-indigo-400 transition-colors">
                  <span className="font-mono text-indigo-600 font-bold block mb-1">/notebooks</span>
                  <p className="text-sm text-gray-500">Jupyter notebooks for EDA and incremental experiments.</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg border border-dashed border-gray-300 hover:border-indigo-400 transition-colors">
                  <span className="font-mono text-indigo-600 font-bold block mb-1">/reports</span>
                  <p className="text-sm text-gray-500">Generated Sweetviz data profiles and SHAP interpretation plots.</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg border border-dashed border-gray-300 hover:border-indigo-400 transition-colors">
                  <span className="font-mono text-indigo-600 font-bold block mb-1">/app</span>
                  <p className="text-sm text-gray-500">The Streamlit application code for user interface.</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg border border-dashed border-gray-300 hover:border-indigo-400 transition-colors">
                  <span className="font-mono text-indigo-600 font-bold block mb-1">/models</span>
                  <p className="text-sm text-gray-500">Serialized .pkl artifacts for the preprocessor and model.</p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'demo' && (
            <div className="max-w-xl mx-auto flex flex-col items-center">
              <div className="w-full border rounded-xl shadow-2xl bg-white overflow-hidden">
                <div className="bg-indigo-50 border-b p-4 text-center">
                  <h3 className="text-lg font-bold text-indigo-700 flex items-center justify-center gap-2">
                    <ShieldCheckIcon size={20} /> Credit Risk Analyzer
                  </h3>
                  <p className="text-[10px] uppercase tracking-widest text-indigo-400 font-bold">Interactive Frontend Mockup</p>
                </div>
                
                <div className="p-6">
                  <form onSubmit={handleAnalyze} className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs font-bold text-gray-500 uppercase mb-1">Monthly Income ($)</label>
                        <input type="number" defaultValue="5000" className="w-full border p-2 rounded focus:ring-2 focus:ring-indigo-500 outline-none" required />
                      </div>
                      <div>
                        <label className="block text-xs font-bold text-gray-500 uppercase mb-1">Loan Amount ($)</label>
                        <input type="number" defaultValue="15000" className="w-full border p-2 rounded focus:ring-2 focus:ring-indigo-500 outline-none" required />
                      </div>
                      <div>
                        <label className="block text-xs font-bold text-gray-500 uppercase mb-1">Credit Score</label>
                        <input type="number" defaultValue="720" max="850" min="300" className="w-full border p-2 rounded focus:ring-2 focus:ring-indigo-500 outline-none" required />
                      </div>
                      <div>
                        <label className="block text-xs font-bold text-gray-500 uppercase mb-1">Age</label>
                        <input type="number" defaultValue="34" className="w-full border p-2 rounded focus:ring-2 focus:ring-indigo-500 outline-none" required />
                      </div>
                    </div>
                    
                    <button 
                      type="submit"
                      disabled={isAnalyzing}
                      className="w-full bg-indigo-600 text-white py-3 rounded-lg font-bold hover:bg-indigo-700 transition flex items-center justify-center gap-2 disabled:opacity-50"
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader2Icon size={20} className="animate-spin" />
                          Running Pipeline...
                        </>
                      ) : 'Analyze Risk Profile'}
                    </button>
                  </form>

                  {/* Dynamic Result Area */}
                  {predictionResult && (
                    <div className={`mt-8 p-6 rounded-xl border-2 animate-in fade-in slide-in-from-bottom-4 duration-500 ${predictionResult.risk === 'High Risk' ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
                       <div className="flex items-start gap-4">
                         {predictionResult.risk === 'High Risk' ? (
                           <AlertTriangleIcon className="text-red-600 shrink-0" size={32} />
                         ) : (
                           <CheckCircle2Icon className="text-green-600 shrink-0" size={32} />
                         )}
                         <div className="flex-1">
                            <h4 className={`text-xl font-bold ${predictionResult.risk === 'High Risk' ? 'text-red-800' : 'text-green-800'}`}>
                              {predictionResult.risk}
                            </h4>
                            <p className="text-sm text-gray-600 mt-1">
                              Probability of default: <span className="font-mono font-bold">{(predictionResult.prob * 100).toFixed(0)}%</span>
                            </p>
                            
                            <div className="mt-4 flex flex-col gap-2">
                              <div className="text-xs font-bold uppercase text-gray-400">Key Risk Drivers (Simulated)</div>
                              <div className="flex flex-wrap gap-2">
                                <span className="bg-white px-2 py-1 rounded border text-xs text-gray-600">Credit Score: 720</span>
                                <span className="bg-white px-2 py-1 rounded border text-xs text-gray-600">DTI: 3.0</span>
                                <span className="bg-white px-2 py-1 rounded border text-xs text-gray-600">Age: 34</span>
                              </div>
                            </div>
                         </div>
                       </div>
                    </div>
                  )}
                </div>
                
                <div className="p-4 bg-gray-50 border-t flex justify-between items-center text-[10px] text-gray-400">
                  <span>ML Artifacts: Loaded (Mock)</span>
                  <span>Model: XGBClassifier v2.0.0</span>
                </div>
              </div>
              <p className="mt-4 text-xs text-gray-400 italic">This is a functional React mockup. Run `streamlit run app/streamlit_app.py` for the real backend-connected experience.</p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
