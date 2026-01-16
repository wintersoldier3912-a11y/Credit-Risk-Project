
import React, { useState } from 'react';
import { 
  FolderIcon, 
  FileIcon, 
  TerminalIcon, 
  LayoutIcon, 
  BookOpenIcon,
  ShieldCheckIcon,
  BarChart3Icon
} from 'lucide-react';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('readme');

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

  return (
    <div className="min-h-screen flex flex-col font-sans text-gray-900">
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
              <div className="grid grid-cols-3 gap-4 mb-6">
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
              <div className="bg-gray-900 text-gray-300 p-4 rounded-lg font-mono text-sm leading-relaxed">
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
              <div className="space-y-4">
                <div className="p-4 bg-gray-50 rounded-lg border border-dashed border-gray-300">
                  <span className="font-mono text-indigo-600 font-bold">/src</span>
                  <p className="text-sm text-gray-500">Core logic for feature engineering, data loading, and model evaluation.</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg border border-dashed border-gray-300">
                  <span className="font-mono text-indigo-600 font-bold">/notebooks</span>
                  <p className="text-sm text-gray-500">Jupyter notebooks for EDA and incremental experiments.</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg border border-dashed border-gray-300">
                  <span className="font-mono text-indigo-600 font-bold">/reports</span>
                  <p className="text-sm text-gray-500">Generated Sweetviz data profiles and SHAP interpretation plots.</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg border border-dashed border-gray-300">
                  <span className="font-mono text-indigo-600 font-bold">/app</span>
                  <p className="text-sm text-gray-500">The Streamlit application code for user interface.</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg border border-dashed border-gray-300">
                  <span className="font-mono text-indigo-600 font-bold">/models</span>
                  <p className="text-sm text-gray-500">Serialized .pkl artifacts for the preprocessor and model.</p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'demo' && (
            <div className="max-w-md mx-auto border rounded-xl p-6 shadow-xl bg-white">
              <div className="text-center mb-6 border-b pb-4">
                <h3 className="text-lg font-bold text-indigo-700">Credit Risk Predictor</h3>
                <p className="text-xs text-gray-400">Streamlit Mockup</p>
              </div>
              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-bold text-gray-500 uppercase">Monthly Income</label>
                  <input type="number" className="w-full border p-2 rounded bg-gray-50" placeholder="5000" />
                </div>
                <div>
                  <label className="block text-xs font-bold text-gray-500 uppercase">Credit Score</label>
                  <input type="number" className="w-full border p-2 rounded bg-gray-50" placeholder="720" />
                </div>
                <button className="w-full bg-indigo-600 text-white py-3 rounded-lg font-bold hover:bg-indigo-700 transition">
                  Analyze Risk
                </button>
                <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg hidden">
                   <div className="text-red-700 font-bold">Result: High Risk (0.87)</div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
