import { useState } from 'react'

// Change this after you deploy the backend (Day 3 deployment step)
const API_URL = 'http://127.0.0.1:8000'

const GRADE_COLORS = {
  A: '#2F9E6E',
  B: '#7CB342',
  C: '#E8A33D',
  D: '#E8590C',
  E: '#E5484D',
}

const FIELD_GROUPS = [
  {
    title: '01 — Income & Household',
    fields: [
      { key: 'age', label: 'Age', default: 35, step: 1 },
      { key: 'MonthlyIncome', label: 'Monthly income ($)', default: 5000, step: 100 },
      { key: 'NumberOfDependents', label: 'Dependents', default: 1, step: 1 },
    ],
  },
  {
    title: '02 — Credit Utilization',
    fields: [
      { key: 'RevolvingUtilizationOfUnsecuredLines', label: 'Revolving utilization (0–2)', default: 0.45, step: 0.05 },
      { key: 'DebtRatio', label: 'Debt ratio', default: 0.3, step: 0.05 },
      { key: 'NumberOfOpenCreditLinesAndLoans', label: 'Open credit lines & loans', default: 6, step: 1 },
      { key: 'NumberRealEstateLoansOrLines', label: 'Real estate loans/lines', default: 1, step: 1 },
    ],
  },
  {
    title: '03 — Payment History',
    fields: [
      { key: 'NumberOfTime30-59DaysPastDueNotWorse', label: '30–59 days past due (count)', default: 0, step: 1 },
      { key: 'NumberOfTime60-89DaysPastDueNotWorse', label: '60–89 days past due (count)', default: 0, step: 1 },
      { key: 'NumberOfTimes90DaysLate', label: '90+ days late (count)', default: 0, step: 1 },
    ],
  },
]

function defaultFormState() {
  const state = {}
  FIELD_GROUPS.forEach((group) => group.fields.forEach((f) => (state[f.key] = f.default)))
  return state
}

export default function App() {
  const [form, setForm] = useState(defaultFormState())
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value === '' ? '' : Number(value) }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!res.ok) throw new Error(`Request failed (${res.status})`)
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError('Could not reach the API. Is uvicorn running on ' + API_URL + '?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <span className="eyebrow">Explainable Underwriting Demo</span>
        <h1>RiskPilot</h1>
        <p className="subhead">
          Every score below comes with the specific factors that drove it — the same standard
          real lenders are held to under fair-lending disclosure rules.
        </p>
      </header>

      <div className="layout">
        <form className="panel form-panel" onSubmit={handleSubmit}>
          {FIELD_GROUPS.map((group) => (
            <fieldset key={group.title}>
              <legend>{group.title}</legend>
              {group.fields.map((f) => (
                <label key={f.key} className="field">
                  <span>{f.label}</span>
                  <input
                    type="number"
                    step={f.step}
                    value={form[f.key]}
                    onChange={(e) => handleChange(f.key, e.target.value)}
                    required
                  />
                </label>
              ))}
            </fieldset>
          ))}
          <button type="submit" disabled={loading}>
            {loading ? 'Scoring…' : 'Assess applicant'}
          </button>
          {error && <p className="error">{error}</p>}
        </form>

        <div className="panel result-panel">
          {!result && !loading && (
            <div className="empty-state">
              <p>Fill in the applicant details and click "Assess applicant."</p>
              <p className="empty-sub">The result will show a risk grade and the exact factors behind it.</p>
            </div>
          )}

          {loading && <div className="empty-state">Scoring applicant…</div>}

          {result && (
            <div className="result">
              <div
                className="stamp"
                style={{ '--grade-color': GRADE_COLORS[result.risk_grade] }}
              >
                <span className="stamp-grade">{result.risk_grade}</span>
                <span className="stamp-decision">{result.decision}</span>
              </div>

              <div className="probability">
                <span className="probability-label">Default probability</span>
                <span className="probability-value">
                  {(result.risk_probability * 100).toFixed(1)}%
                </span>
              </div>

              <div className="factors">
                <span className="factors-label">Top factors behind this score</span>
                {result.top_factors.map((f) => {
                  const isRisk = f.shap_value > 0
                  const magnitude = Math.min(Math.abs(f.shap_value) * 40, 100)
                  return (
                    <div className="factor-row" key={f.feature}>
                      <div className="factor-row-top">
                        <span className="factor-name">{f.feature}</span>
                        <span className={`factor-tag ${isRisk ? 'up' : 'down'}`}>
                          {isRisk ? '↑ risk' : '↓ risk'}
                        </span>
                      </div>
                      <div className="factor-bar-track">
                        <div
                          className={`factor-bar ${isRisk ? 'up' : 'down'}`}
                          style={{ width: `${magnitude}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
